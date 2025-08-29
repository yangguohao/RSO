import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import torch.utils.data
import torchvision.models
from datetime import datetime
from torch.autograd.profiler import record_function
from transformers import AutoModelForSequenceClassification
from optimizer_cp import AdamW, Flora
from models_cp import create_RSO_model, create_peft_model


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
            # timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            file_name = f"./mem_results/visual_mem_flora"

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

        class args:
            lora_r = 4
            lora_alpha = 8
            lora_dropout = 0
            device = "cuda:0"
            rslora = False

        model.to(device)
        # model = create_RSO_model(model, args)
        # model = create_peft_model(model, args)

        x = torch.randint(0, 128, size=(32, 128), device=device)
        model.train()
        labels = torch.randint(0, 2, size=(32,), device=device)
        # optimizer = torch.optim.AdamW(model.parameters())
        # optimizer = AdamW(model.named_parameters(), args=args)
        compress_params = []
        compress_names = []
        params = []
        names = []
        for k, v in model.named_parameters():
            if any(x in k for x in ["query", "value", "key",
                                    "intermediate.dense", "output.dense"]) and v.ndim == 2:
                compress_params.append(v)
                compress_names.append(k)

            else:
                params.append(v)
                names.append(k)

        optimizer = Flora([
            {"params": params,
             "names": names},
            {
                "params": compress_params,
                "names": compress_names,
                "rank": args.lora_r,
            },
        ], args=args)
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
