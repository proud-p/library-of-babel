import torch
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from osc_server import HandCoordReceiver  # Reuse the clean, isolated OSC listener
from osc_client import OSCSender

# === GPT-2 Setup ===
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
prompt = "The meaning of life is"
output = generator(prompt, max_length=50, num_return_sequences=2)
text1 = output[0]["generated_text"]
text2 = output[1]["generated_text"]

print("\n📝 GPT-2 Generated Answer 1:\n", text1)
print("\n📝 GPT-2 Generated Answer 2:\n", text2)

# === Convert to token embeddings ===
tokens1 = tokenizer(text1, return_tensors="pt")["input_ids"].to(model.device)
tokens2 = tokenizer(text2, return_tensors="pt")["input_ids"].to(model.device)

with torch.no_grad():
    embedding1 = model.get_input_embeddings()(tokens1).squeeze(0)
    embedding2 = model.get_input_embeddings()(tokens2).squeeze(0)

max_len = max(embedding1.size(0), embedding2.size(0))
embedding1 = torch.nn.functional.pad(embedding1, (0, 0, 0, max_len - embedding1.size(0)))
embedding2 = torch.nn.functional.pad(embedding2, (0, 0, 0, max_len - embedding2.size(0)))

print("\n✅ Extracted Embeddings Shape:", embedding1.shape)

# === Interpolation Function ===
def get_answer(v0, v1, coord_x=0.0, coord_y=0.0):
    def noise_mult(t):
        return np.exp(-((t - 0.5) / 0.2) ** 2)

    # seed = int(coord_x * 1000) + int(coord_y * 1000) * 10000
    seed = 42
    torch.manual_seed(seed)
    noise_x = torch.randn_like(v0).to(model.device)
    torch.manual_seed(seed + 1)
    noise_y = torch.randn_like(v0).to(model.device)

    dot = torch.sum(v0 * v1, axis=-1) / (torch.linalg.norm(v0, axis=-1) * torch.linalg.norm(v1, axis=-1))
    dot = torch.clamp(dot, -1.0, 1.0)
    theta = torch.arccos(dot)
    sin_theta = torch.sin(theta)

    nm = noise_mult(coord_x)
    v = ((torch.sin((1 - coord_x) * theta) / sin_theta)[:, None] * v0 + 
         (torch.sin(coord_x * theta) / sin_theta)[:, None] * v1 + 
         noise_x * nm *coord_x + noise_y * nm *coord_y)

    return torch.tensor(v, dtype=torch.float32)

# === Custom OSC-Integrated Receiver Class ===
class GPTFromOSC(HandCoordReceiver):
    def __init__(self, sender_ip, sender_port):
        super().__init__()
        self.sender = OSCSender(ip=sender_ip, port=sender_port)

    def handle_xyz(self, address, *args):
        if len(args) == 3:
            self.latest_coords["x"] = round(args[0], 1)
            self.latest_coords["y"] = round(args[1], 1)
            self.latest_coords["z"] = round(args[2], 1)

            x, y = self.latest_coords["x"], self.latest_coords["y"]
            print(f"\n📥 Received /xyz: x={x}, y={y}")

            latent = get_answer(embedding1, embedding2, coord_x=x, coord_y=y)
            with torch.no_grad():
                logits = model.lm_head(latent)
                ids = torch.argmax(logits, dim=-1)
                decoded = tokenizer.decode(ids, skip_special_tokens=True)

            print(f"🧠 GPT Response for x={x:.2f}, y={y:.2f}: {decoded}")

            # Send response via OSC
            self.sender.send_osc_message(decoded)

# === Main Entry ===
if __name__ == "__main__":
    receiver_ip = "0.0.0.0"      # This machine (WSL) 
    receiver_port = 5009
    # sender_ip = "192.168.0.2"    # Windows machine -James' house
    sender_ip = "10.106.32.181" # Windows machine
    sender_port = 1234

    gpt_osc = GPTFromOSC(sender_ip, sender_port)
    gpt_osc.start_receiver(ip=receiver_ip, port=receiver_port)
