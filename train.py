import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.transformer_post import Transformer
from dataloader_pt import TranslationDataset, TranslationCollator
from datasets import load_dataset
from transformers import AutoTokenizer
import wandb
import time
from bleu import compute_bleu
import matplotlib.pyplot as plt
import torch.nn as nn
import os
from model.dyt import DyT

# 토크나이저 로드 (영어 -> 독일어 번역 예시)
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")


# 모델 하이퍼파라미터 설정
input_size = len(tokenizer) + 1
hidden_size = 512
output_size = len(tokenizer) + 1
n_splits = 8
n_enc_blocks = 6
n_dec_blocks = 6
dropout_p = 0.1
max_length = 128  # 긴 문장이 있을 수 있으므로 적절한 길이로 설정
batch_size = 150
epochs = 200
patience = 3


# wandb 연결
wandb.init(project="translation", config={
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "n_splits": n_splits,
    "n_enc_blocks": n_enc_blocks,
    "n_dec_blocks": n_dec_blocks,
    "dropout_p": dropout_p,
    "max_length": max_length,
    "batch_size": batch_size,
    "epochs": epochs,
    "patience": patience
}
# resume='allow',
# id='5mk8s55t'
)

# # wandb config를 변수에 할당
config = wandb.config


# 데이터셋 및 데이터로더 설정
train_dataset = TranslationDataset(split='train')
valid_dataset = TranslationDataset(split='validation')


collate_fn = TranslationCollator(tokenizer,  max_length=max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=50, shuffle=False, collate_fn=collate_fn)

### Layernorm확인
ln_activations = {}
def hook_fn(module, input, output):
    """LayerNorm의 input과 output을 저장하는 Hook"""
    ln_activations[module] = (input[0].detach().cpu(), output.detach().cpu())

# LayerNorm에 Hook을 등록하는 함수
def register_hooks(model):
    for name, module in model.named_modules():
        if isinstance(module, DyT):
            module.register_forward_hook(hook_fn)



def init_weights(m):
    if isinstance(m, nn.Linear):
        # ReLU 활성화 함수에 맞게 He 초기화 적용
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# 모델 초기화
model = Transformer(input_size, hidden_size, output_size, n_splits,  n_enc_blocks, n_dec_blocks, dropout_p, max_length)
model.apply(init_weights)
model.load_state_dict(torch.load("/home/jovyan/transformer_Post_best.pth"))
register_hooks(model) # hook

# 옵티마이저와 손실 함수 설정
optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
criterion = torch.nn.NLLLoss(ignore_index=tokenizer.pad_token_id)




# 학습 함수
def train(model, train_loader, optimizer, scheduler, criterion, device):
    model.train()
    running_loss = 0.0
    loop = tqdm(train_loader, total=len(train_loader), desc="Training")
    start_time = time.time()

    for batch in loop:
        optimizer.zero_grad()

        # 데이터 준비
        src = batch['input_ids'].to(device)
        tgt = batch['tgt_ids'].to(device)
        label = batch['labels'].to(device)
        attention_mask_src = batch['attention_mask_src'].to(device)

        # 모델 출력 계산
        output = model(src, tgt, attention_mask_src) 
        output = output.view(-1, output.shape[-1])  # (bs * seq_len, vocab_size)
        
        label = label.contiguous().view(-1) 
        loss = criterion(output, label)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        loop.set_postfix(loss=running_loss / (loop.n + 1))  # 진행 중인 평균 손실 출력

    # 측정
    end_time = time.time()
    gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    gpu_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    epoch_time = end_time - start_time

    wandb.log({"train_gpu_memory": gpu_memory, "train_gpu_reserved": gpu_reserved, "train_time": epoch_time})
    return running_loss / len(train_loader)

# 검증 함수
def validate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    start_time = time.time()
    predictions = []
    refer = []
    

    loop = tqdm(valid_loader, total=len(valid_loader), desc="Validation")
    with torch.no_grad():
        for batch in loop:
            # 데이터 준비
            src = batch['input_ids'].to(device)
            tgt = batch['tgt_ids'].to(device)
            label = batch['labels'].to(device)
            attention_mask_src = batch['attention_mask_src'].to(device)

            # 모델 출력 계산
            output = model(src, tgt, attention_mask_src)
            
        # LayerNorm 결과 시각화 및 저장
            save_dir = "DyT_visualization"
            os.makedirs(save_dir, exist_ok=True)

            for i, (module, (ln_input, ln_output)) in enumerate(ln_activations.items()):
                ln_input = ln_input.view(-1).numpy()  # Flatten
                ln_output = ln_output.view(-1).numpy()  # Flatten

                plt.figure(figsize=(5, 5))
                plt.scatter(ln_input, ln_output, alpha=0.5, s=1)
                plt.xlabel("DyT input")
                plt.ylabel("DyT output")
                plt.title(f"DyT Output vs Input\n{module}")

                # 파일 저장
                save_path = os.path.join(save_dir, f"DyT_scatter_{i}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()  # 메모리 절약

            #     print(f"LayerNorm 그래프 저장 완료: {save_path}")
            #     breakpoint()
        
            output_flat = output.view(-1, output.shape[-1])  # (bs * seq_len, vocab_size)
            label_flat = label.contiguous().view(-1)

            # 손실 계산
            loss = criterion(output_flat, label_flat)

            running_loss += loss.item()

            loop.set_postfix(loss=running_loss / (loop.n + 1))  # 진행 중인 평균 손실 출력


            # bleu
            # 각 단어에의 vocab에서 가장 높은 idx를 선택
            predicted_tokens = output.argmax(dim=-1)   #(bs, seq_len)
        
            for i in range(predicted_tokens.shape[0]):
                references = []
                # 문장마다 bs마다 디코드를 한다
                pred_sentence = tokenizer.decode(predicted_tokens[i].cpu().numpy(), skip_special_tokens=True)
         
                ref_sentence = tokenizer.decode(label[i].cpu().numpy(), skip_special_tokens=True)

                predictions.append(pred_sentence.split())  # BLEU 계산을 위해 토큰화된 문장 추가
                references.append(ref_sentence.split())  # 여러 개의 참조 번역이 가능하므로 리스트로 감싸기
                refer.append(references)
               
    # 측정
    end_time = time.time()
    gpu_memory = torch.cuda.memory_allocated(device) / (1024 ** 2)
    gpu_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    epoch_time = end_time - start_time

    bleu_score = compute_bleu(refer, predictions)
    wandb.log({"val_gpu_memory": gpu_memory, "val_gpu_reserved": gpu_reserved, "val_time": epoch_time})
    return running_loss / len(valid_loader), bleu_score

# 학습 실행
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # 학습 단계
    train_loss = train(model, train_loader, optimizer, scheduler, criterion, device)
    print(f"Training Loss: {train_loss:.4f}")
    
    # 검증 단계
    val_loss, bleu_score = validate(model, valid_loader, criterion, device)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation BLEU Score: {bleu_score:.4f}")
    
    current_lr = scheduler.get_last_lr()[0]
    # wandb에 학습 및 검증 손실 기록
    wandb.log({"train_loss": train_loss, "val_loss": val_loss, "bleu_score": bleu_score, "learning_rate_epoch_end": current_lr})

    # 검증 손실이 개선되었으면 모델 저장
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "transformer_model_best.pth")
        print("Model saved!")
        patience_counter = 0
    else:
        patience_counter +=1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch + 1}. No improvement in validation loss for {patience} epochs.")
        break

# 최종 모델 저장
torch.save(model.state_dict(), "transformer_model_final.pth")
wandb.finish()