import os
import sys
import logging
import traceback
import datasets
import transformers
import modelscope
import torch
from transformers import TrainerCallback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)


class VerboseTrainingCallback(TrainerCallback):
    """自定义回调函数，提供更详细的训练进度信息"""

    def __init__(self):
        self.train_loss = []

    def on_train_begin(self, args, state, control, **kwargs):
        logging.info("🚀 训练开始")
        self.train_loss = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        logging.info(f"📘 开始第 {state.epoch + 1} 轮训练")

    def on_step_end(self, args, state, control, **kwargs):
        # 检查是否需要记录日志
        if state.global_step % args.logging_steps == 0:
            # 安全获取最近的损失值
            if state.log_history and 'loss' in state.log_history[-1]:
                current_loss = state.log_history[-1]['loss']
                self.train_loss.append(current_loss)
                logging.info(f"🔢 训练进度: 步数 {state.global_step}, 损失 {current_loss:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        logging.info("✅ 训练完成")
        logging.info(f"训练损失记录: {self.train_loss}")


def check_cuda_availability():
    """检查CUDA可用性"""
    logging.info("检查GPU可用性...")
    logging.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA设备数量: {torch.cuda.device_count()}")
        logging.info(f"当前CUDA设备: {torch.cuda.current_device()}")
        logging.info(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")


def main():
    try:
        # 检查GPU
        check_cuda_availability()

        # 创建必要目录
        os.makedirs("./data", exist_ok=True)
        os.makedirs("./LLM05", exist_ok=True)
        os.makedirs("./logs", exist_ok=True)

        # 加载数据集
        logging.info("1. 加载数据集中...")
        try:
            raw_datasets = datasets.load_dataset(
                "json",
                data_files="./data/wikipedia-cn-20230720-filtered.json"
            )
        except Exception as e:
            logging.error(f"数据集加载失败: {e}")
            raise

        raw_datasets = raw_datasets["train"].train_test_split(test_size=0.1, seed=2222)
        logging.info(f"训练集大小: {len(raw_datasets['train'])}")
        logging.info(f"验证集大小: {len(raw_datasets['test'])}")

        # 下载并保存tokenizer和配置
        logging.info("2. 下载并保存模型配置与分词器...")
        modelscope.AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B").save_pretrained(
            "Qwen2.5-0.5B"
        )
        modelscope.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B").save_pretrained(
            "Qwen2.5-0.5B"
        )

        context_length = 512
        tokenizer = transformers.AutoTokenizer.from_pretrained("./Qwen2.5-0.5B")

        # 数据预处理
        def tokenize(element):
            outputs = tokenizer(
                element["completion"],
                truncation=True,
                max_length=context_length,
                return_overflowing_tokens=True,
                return_length=True,
            )
            input_batch = [
                input_ids for length, input_ids in zip(outputs["length"], outputs["input_ids"])
                if length == context_length
            ]
            return {"input_ids": input_batch}

        logging.info("3. 分词数据集...")
        tokenized_datasets = raw_datasets.map(
            tokenize,
            batched=True,
            remove_columns=raw_datasets["train"].column_names
        )

        tokenizer.pad_token = tokenizer.eos_token
        data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)

        # 模型配置
        logging.info("4. 准备模型配置...")
        config = transformers.AutoConfig.from_pretrained(
            "./Qwen2.5-0.5B",
            vocab_size=len(tokenizer),
            hidden_size=512,
            intermediate_size=2048,
            num_attention_heads=8,
            num_hidden_layers=12,
            n_ctx=context_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = transformers.Qwen2ForCausalLM(config)
        model_size = sum(t.numel() for t in model.parameters())

        logging.info(f"模型大小: {model_size / 1000 ** 2:.1f}M 参数")

        # 训练参数
        logging.info("5. 配置训练参数...")
        args = transformers.TrainingArguments(
            output_dir="LLM05",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            eval_strategy="steps",
            eval_steps=500,
            logging_steps=50,
            gradient_accumulation_steps=8,
            num_train_epochs=5,
            weight_decay=0.1,
            warmup_steps=200,
            optim="adamw_torch",
            lr_scheduler_type="cosine",
            learning_rate=2e-5,
            save_steps=500,
            save_total_limit=10,
            bf16=True if torch.cuda.is_available() else False,
            logging_dir='./logs'
        )

        # 初始化训练器
        logging.info("6. 初始化训练器...")
        verbose_callback = VerboseTrainingCallback()
        trainer = transformers.Trainer(
            model=model,
            tokenizer=tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["test"],
            callbacks=[verbose_callback]
        )

        # 开始训练
        logging.info("🚀 开始模型训练...")
        trainer.train()

        # 保存模型
        logging.info("7. 保存模型...")
        model.save_pretrained("./LLM05/Weight")
        tokenizer.save_pretrained("./LLM05/Weight")

    except Exception as e:
        logging.error(f"训练过程中发生严重错误: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()