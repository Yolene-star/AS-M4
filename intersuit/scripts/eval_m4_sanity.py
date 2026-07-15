#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from intersuit.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from intersuit.conversation import conv_templates, SeparatorStyle
from intersuit.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, process_images, tokenizer_image_token
from intersuit.model.builder import load_pretrained_model


def build_prompt(question, conv_mode):
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv, conv.get_prompt()


def run_case(model_path, prompt, image_path=None, max_new_tokens=128, device="cuda:0"):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        device_map=device,
        multimodal=True,
        attn_implementation="eager",
        overwrite_config={
            "mm_spatial_pool_stride": 2,
            "mm_spatial_pool_mode": "average",
        },
    )
    model.eval()
    model.to(device)

    conv_mode = "qwen_1_5" if "qwen" in model_path.lower() else "llava_v1"
    if image_path:
        question = prompt if DEFAULT_IMAGE_TOKEN in prompt else DEFAULT_IMAGE_TOKEN + "\n" + prompt
    else:
        question = prompt
    conv, full_prompt = build_prompt(question, conv_mode)

    input_ids = tokenizer_image_token(full_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids, device=device)

    images = None
    image_sizes = None
    modalities = None
    if image_path:
        image = Image.open(image_path).convert("RGB")
        image_sizes = [image.size]
        images = process_images([image], image_processor, model.config)
        model_dtype = next(model.parameters()).dtype
        if isinstance(images, list):
            images = [img.to(device=device, dtype=model_dtype) for img in images]
        else:
            images = images.to(device=device, dtype=model_dtype)
        modalities = ["image"]

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            images=images,
            image_sizes=image_sizes,
            modalities=modalities,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    text = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return text


def main():
    parser = argparse.ArgumentParser(description="M4 checkpoint sanity inference.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image_path")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_json")
    args = parser.parse_args()

    output = run_case(
        model_path=args.model_path,
        prompt=args.prompt,
        image_path=args.image_path,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
    )
    result = {
        "model_path": args.model_path,
        "prompt": args.prompt,
        "image_path": args.image_path,
        "output": output,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
