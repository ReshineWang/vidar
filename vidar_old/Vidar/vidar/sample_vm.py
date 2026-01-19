import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from vm.utils.file_utils import save_videos_grid
from vm.config import parse_args
from vm.inference import HunyuanVideoSampler

import json
import cv2
from PIL import Image

def get_prompt_and_image(json_path,save_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    video_path = data['video_path']
    prompt = data['raw_caption']['long caption']

    task_name = Path(video_path).parent.name + "_" + os.path.basename(video_path).replace('mp4','')
    # get image of first frame of video
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()

    # store img
    dir = save_dir
    os.makedirs(dir,exist_ok=True)
    image_path = os.path.join(dir, os.path.basename(video_path).replace('mp4','jpg'))
    cv2.imwrite(image_path, frame)
    cap.release()

    return task_name,image_path,prompt

def main():
    args = parse_args()
    print(args)

    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    image_paths = []
    prompts = []
    tasks = []

    if args.metadata_paths is None:
        for prompt,image_path in zip(args.prompt,args.i2v_images_path):
            image_paths.append(image_path)
            prompts.append(prompt)
            tasks.append(Path(image_path).name)
    else:
        for json_path in args.metadata_paths:
            taskname,image_path, prompt = get_prompt_and_image(json_path,args.image_save_dir)
            image_paths.append(image_path)
            prompts.append(prompt)
            tasks.append(taskname)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    for image_path, prompt in zip(image_paths,prompts):
        print("get image:",image_path)

        i2v_images = [Image.open(image_path).convert('RGB')]
        # TODO: batch inference check
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            i2v_mode=args.i2v_mode,
            i2v_resolution=args.i2v_resolution,
            i2v_images=i2v_images,
            i2v_condition_type=args.i2v_condition_type,
            i2v_stability=args.i2v_stability,
            ulysses_degree=args.ulysses_degree,
            ring_degree=args.ring_degree,
        )
        samples = outputs['samples']

        # sample size (1,3,61,560,512)

        # Save samples
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H-%M-%S")

                # print("image path:",image_path)
                has_stability = "i2v_stability" if args.i2v_stability else "" 
                # 20xx-xx-xx-xx:xx:xx_seed0_flow_shiftxx_cfg_scalexx_infer_stepxx_prompt[-100:]

                video_name = tasks[i] + '_'+os.path.basename(image_path).replace('.jpg','.mp4')
                cur_save_path = f"{save_path}/{time_flag}_{video_name}"
                # _{outputs['prompts'][i][-100:].replace('/','')}.mp4"
                                
                save_videos_grid(sample, cur_save_path, fps=24)
                logger.info(f'Sample save to: {cur_save_path}')

if __name__ == "__main__":
    main()
