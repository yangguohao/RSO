import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import torch.utils.data
import torchvision.models
from datetime import datetime
from torch.autograd.profiler import record_function
from transformers import AutoModelForSequenceClassification


def test_transformer():
    def train(num_iter=5, device="cuda:0"):
        model = nn.Transformer(d_model=512, nhead=2, num_encoder_layers=2, num_decoder_layers=2).to(device=device)
        x = torch.randn(size=(1, 1024, 512), device=device)
        tgt = torch.rand(size=(1, 1024, 512), device=device)
        model.train()
        labels = torch.rand_like(model(x, tgt))
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        for _ in range(num_iter):
            y = model(x, tgt)
            loss = criterion(y, labels)
            loss.backward()
            print(loss.item())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    def run():
        # Start recording memory snapshot history
        torch.cuda.memory._record_memory_history(max_entries=100000)

        # training running:
        train()

        timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
        file_name = f"visual_mem_{timestamp}.pickle"
        # save record:
        torch.cuda.memory._dump_snapshot(file_name)

        # Stop recording memory snapshot history:
        torch.cuda.memory._record_memory_history(enabled=None)

    if __name__ == "__main__":
        run()


def test_profiler():
    def train(num_iter=5, device="cuda:0", warmup=False):
        def trace_handler(prof: torch.profiler.profile):
            # Prefix for file names.
            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            file_name = f"./visual_mem"

            # Construct the trace file.
            # prof.export_chrome_trace(f"{file_name}.json.gz")

            # Construct the memory timeline file.
            if not warmup:
                prof.export_memory_timeline(f"{file_name}.html", device=device)

        model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-large",
            num_labels=2
            # config=RobertaConfig.from_pretrained(
            #     "roberta-base",
            #     attention_probs_dropout_prob=0.0,
            #     hidden_dropout_prob=0.0,
            #     classifier_dropout=0.0,  # 有的任务模型支持这个
            #     num_labels=num_labels,
            # )
        )
        model.to(device)
        x = torch.randint(0, 128, size=(32, 128), device=device)
        model.train()
        labels = torch.randint(0, 2, size=(32,), device=device)
        optimizer = torch.optim.AdamW(model.parameters())
        with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                on_trace_ready=trace_handler,
        ) as prof:
            for _ in range(num_iter):
                prof.step()
                with record_function("## forward ##"):
                    loss = model(x, labels=labels).loss

                with record_function("## backward ##"):
                    loss.backward()
                    print(loss.item())

                with record_function("## optimizer ##"):
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

    if __name__ == "__main__":
        # warm-up:
        train(1, warmup=True)
        # run:
        train(3)


# use_snapshot()
# test_transformer()
test_profiler()
