# walk between 2 comprehensible points, linear interpolation with some noise

#circular walk

import torch
import numpy as np
from tqdm import tqdm
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from get_hand_coords import get_coord
import json

global device
ip = "192.168.0.2"
print(f"Serving on ip {ip}, IS THIS RIGHT? CHECK")

# Load GPT-2 with tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

# Initialize text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
)

# 🔹 Generate an initial GPT-2 response
prompt = "The meaning of life is" #TODO this should be an input, read from website?
output = generator(prompt, max_length=50, num_return_sequences=2)
generated_text_1 = output[0]["generated_text"]
generated_text_2 = output[1]["generated_text"]
print("\n📝 GPT-2 Generated Answer 1:\n", generated_text_1)
print("\n📝 GPT-2 Generated Answer 2:\n", generated_text_2)

# 🔹 Convert the generated answer to token embeddings
tokens1 = tokenizer(generated_text_1, return_tensors="pt")["input_ids"].to(model.device)
tokens2 = tokenizer(generated_text_2, return_tensors="pt")["input_ids"].to(model.device)

with torch.no_grad():
    embedding1 = model.get_input_embeddings()(tokens1).squeeze(0)  # Word embeddings
    embedding2 = model.get_input_embeddings()(tokens2).squeeze(0)  # Word embeddings
    
# Ensure embeddings have the same length (pad if necessary)
max_len = max(embedding1.shape[0], embedding2.shape[0])
embedding1 = torch.nn.functional.pad(embedding1, (0, 0, 0, max_len - embedding1.shape[0]))
embedding2 = torch.nn.functional.pad(embedding2, (0, 0, 0, max_len - embedding2.shape[0]))

print("\n✅ Extracted Embeddings Shape:", embedding1.shape)

# 🔹 Generate a loop interpolation in latent space
num_interpolation_steps = 10  # Number of steps for circular walk

# Define LERP function (Linear Interpolation)
# change num steps 
def get_answer(v0, v1,num_steps=10, coord_x=0, coord_y=0):
    
    def noise_mult(t):
        # print("t",t)
        # return min(t,1-t)
        return np.exp(-((t - 0.5) / 0.2) ** 2) #gaussian
        
    # TODO add y in here somewhere and clean this up so it makes more snse
    # Generate two random latent vectors
    noise_x = torch.randn_like(v0).to(model.device)
    noise_y = torch.randn_like(v0).to(model.device)
    
    v0, v1 = v0.to(model.device), v1.to(model.device)
    dot = torch.sum(v0 * v1, axis=-1) / (torch.linalg.norm(v0, axis=-1) * torch.linalg.norm(v1, axis=-1))
    dot = torch.clip(dot, -1.0, 1.0)
    theta = torch.arccos(dot)
    sin_theta = torch.sin(theta)


    nm = noise_mult(coord_x)
    v = ((torch.sin((1 - coord_x) * theta) / sin_theta)[:, None] * v0 + (torch.sin(coord_x * theta) / sin_theta)[:, None] * v1)+ (noise_x*nm) + (noise_y*nm)
    # Add small random noise

    return torch.tensor(v, dtype=torch.float32)

#alternative to noise lerp, walk in curve, might make more sense and seem less random. so if you stop at the same place you should get the same answer
def get_curved_answer(v0, v1, coord_x=0.0, coord_y=0.0):
    """
    Deterministic curved interpolation (like a Bézier) between v0 and v1.
    coord_x: how far along the interpolation we are (0 to 1)
    coord_y: how far to curve (negative = curve one side, positive = the other)
    """
    v0, v1 = v0.to(model.device), v1.to(model.device)

    # Direction from v0 to v1
    direction = v1 - v0
    midpoint = (v0 + v1) / 2

    # Create a stable perpendicular-ish direction using a fixed seed (based on v0 + v1)
    torch.manual_seed(42)  # Ensures this is always the same
    fake_vec = torch.ones_like(direction)
    perp = fake_vec - (fake_vec * direction).sum(-1, keepdim=True) / (direction.norm(dim=-1, keepdim=True) ** 2 + 1e-8) * direction
    perp = torch.nn.functional.normalize(perp, dim=-1)

    # Scale perpendicular offset based on coord_y
    curve_strength = coord_y * 10.0  # Can be positive or negative
    control_point = midpoint + curve_strength * perp  # Curve away from center line

    # Bézier interpolation
    t = torch.tensor(coord_x).float().to(model.device)
    curved = (1 - t)**2 * v0 + 2 * (1 - t) * t * control_point + t**2 * v1
    return curved


# Generate interpolated latents
num_steps = 50


# Decode the interpolated embeddings into words
decoded_sentences = []
while True:
    
    x,y,z = get_coord(ip)
    
        
    # latent = get_answer(embedding1, embedding2, num_steps=num_steps, coord_x=x, coord_y=0.5) 
    latent = get_curved_answer(embedding1, embedding2, coord_x=x, coord_y=(y - 0.5) * 2)  # remap y from [0,1] → [-1,1]



    with torch.no_grad():
        token_logits = model.lm_head(latent)  
        token_ids = torch.argmax(token_logits, dim=-1) 
        decoded_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    
    decoded_sentences.append(decoded_text)
    print(f"Coord {x}: {decoded_text}")