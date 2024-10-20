# coding: utf-8

import os
import gradio as gr
import random
import cv2
import re
import uuid
import base64
import requests
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import argparse
import inspect
import subprocess
from PIL import Image
import numpy as np
from tqdm.auto import tqdm

import torch
from torch import autocast
import torch.utils.checkpoint

from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.memory import ConversationTokenBufferMemory
from langchain.chat_models import ChatOpenAI
from openai import OpenAI as dalle


# Grounding DINO
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from segment_anything import build_sam, SamPredictor, SamAutomaticMaskGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import wget


# OPEN AIのAPIを入力
API_KEY = ""

VISUAL_CHATGPT_PREFIX = """Visual ChatGPT is designed to be able to assist with a wide range of text and visual related tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. Visual ChatGPT is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

Visual ChatGPT is able to process and understand large amounts of text and images. As a language model, Visual ChatGPT can not directly read images, but it has a list of tools to finish different visual tasks. Each image will have a file name formed as "image/xxx.png", and Visual ChatGPT can invoke different tools to indirectly understand pictures. When talking about images, Visual ChatGPT is very strict to the file name and will never fabricate nonexistent files. When using tools to generate new image files, Visual ChatGPT is also known that the image may not be the same as the user's demand, and will use other visual question answering tools or description tools to observe the real image. Visual ChatGPT is able to use tools in a sequence, and is loyal to the tool observation outputs rather than faking the image content and image file name. It will remember to provide the file name from the last tool observation, if a new image is generated.

Human may provide new figures to Visual ChatGPT with a description. The description helps Visual ChatGPT to understand this image, but Visual ChatGPT should use tools to finish following tasks, rather than directly imagine from the description.

Overall, Visual ChatGPT is a powerful visual dialogue assistant tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. 


TOOLS:
------

Visual ChatGPT  has access to the following tools:"""

VISUAL_CHATGPT_FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

VISUAL_CHATGPT_SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
Since Visual ChatGPT is a text language model, Visual ChatGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for Visual ChatGPT, Visual ChatGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""

os.makedirs('image', exist_ok=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def blend_gt2pt(old_image, new_image, sigma=0.15, steps=100):
    new_size = new_image.size
    old_size = old_image.size
    easy_img = np.array(new_image)
    gt_img_array = np.array(old_image)
    pos_w = (new_size[0] - old_size[0]) // 2
    pos_h = (new_size[1] - old_size[1]) // 2

    kernel_h = cv2.getGaussianKernel(old_size[1], old_size[1] * sigma)
    kernel_w = cv2.getGaussianKernel(old_size[0], old_size[0] * sigma)
    kernel = np.multiply(kernel_h, np.transpose(kernel_w))

    kernel[steps:-steps, steps:-steps] = 1
    kernel[:steps, :steps] = kernel[:steps, :steps] / kernel[steps - 1, steps - 1]
    kernel[:steps, -steps:] = kernel[:steps, -steps:] / kernel[steps - 1, -(steps)]
    kernel[-steps:, :steps] = kernel[-steps:, :steps] / kernel[-steps, steps - 1]
    kernel[-steps:, -steps:] = kernel[-steps:, -steps:] / kernel[-steps, -steps]
    kernel = np.expand_dims(kernel, 2)
    kernel = np.repeat(kernel, 3, 2)

    weight = np.linspace(0, 1, steps)
    top = np.expand_dims(weight, 1)
    top = np.repeat(top, old_size[0] - 2 * steps, 1)
    top = np.expand_dims(top, 2)
    top = np.repeat(top, 3, 2)

    weight = np.linspace(1, 0, steps)
    down = np.expand_dims(weight, 1)
    down = np.repeat(down, old_size[0] - 2 * steps, 1)
    down = np.expand_dims(down, 2)
    down = np.repeat(down, 3, 2)

    weight = np.linspace(0, 1, steps)
    left = np.expand_dims(weight, 0)
    left = np.repeat(left, old_size[1] - 2 * steps, 0)
    left = np.expand_dims(left, 2)
    left = np.repeat(left, 3, 2)

    weight = np.linspace(1, 0, steps)
    right = np.expand_dims(weight, 0)
    right = np.repeat(right, old_size[1] - 2 * steps, 0)
    right = np.expand_dims(right, 2)
    right = np.repeat(right, 3, 2)

    kernel[:steps, steps:-steps] = top
    kernel[-steps:, steps:-steps] = down
    kernel[steps:-steps, :steps] = left
    kernel[steps:-steps, -steps:] = right

    pt_gt_img = easy_img[pos_h:pos_h + old_size[1], pos_w:pos_w + old_size[0]]
    gaussian_gt_img = kernel * gt_img_array + (1 - kernel) * pt_gt_img  # gt img with blur img
    gaussian_gt_img = gaussian_gt_img.astype(np.int64)
    easy_img[pos_h:pos_h + old_size[1], pos_w:pos_w + old_size[0]] = gaussian_gt_img
    gaussian_img = Image.fromarray(easy_img)
    return gaussian_img


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    print(f"history_memory:{history_memory}, n_tokens: {n_tokens}")
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


def get_new_image_name(org_img_name, func_name="update"):
    head_tail = os.path.split(org_img_name)
    head = head_tail[0]
    tail = head_tail[1]
    name_split = tail.split('.')[0].split('_')
    this_new_uuid = str(uuid.uuid4())[:4]
    if len(name_split) == 1:
        most_org_file_name = name_split[0]
    else:
        assert len(name_split) == 4
        most_org_file_name = name_split[3]
    recent_prev_file_name = name_split[0]
    new_file_name = f'{this_new_uuid}_{func_name}_{recent_prev_file_name}_{most_org_file_name}.png'
    return os.path.join(head, new_file_name)

class InstructPix2Pix:
    def __init__(self, device):
        print(f"Initializing InstructPix2Pix to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
       
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix",
                                                                           safety_checker=StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker'),
                                                                           torch_dtype=self.torch_dtype).to(device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

    @prompts(name="Instruct Image Using Text",
             description="useful when you want to the style of the image to be like the text. "
                         "like: make it look like a painting. or make it like a robot. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the text. ")
    def inference(self, inputs):
        """Change style of image."""
        print("===>Starting InstructPix2Pix Inference")
        image_path, text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        original_image = Image.open(image_path)
        image = self.pipe(text, image=original_image, num_inference_steps=40, image_guidance_scale=1.2).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="pix2pix")
        image.save(updated_image_path)
        print(f"\nProcessed InstructPix2Pix, Input Image: {image_path}, Instruct Text: {text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class Text2Image:
    def __init__(self, device):
        print(f"Initializing Text2Image not using {device}")
        self.client = dalle(api_key=API_KEY)

    @prompts(name="Generate Image From User Input Text",
             description="useful when you want to generate an image from a user input text and save it to a file. "
                         "like: generate an image of an object or something, or generate an image that includes some objects. "
                         "The input to this tool should be a string, representing the text used to generate image. ")
    def inference(self, text):
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=text,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")

        # OpenAIのAPIから画像をダウンロードして保存
        with open(image_filename, 'wb') as file:
            file.write(requests.get(image_url).content)

        print(f"\nProcessed Text2Image, Input Text: {text}, Output Image: {image_filename}")
        return image_filename


class ImageCaptioning:
    def __init__(self, device):
        print(f"Initializing ImageCaptioning not using {device}")

    @prompts(name="Get Photo Description",
             description="useful when you want to know what is inside the photo. receives image_path as input. "
                         "The input to this tool should be a string, representing the image_path. ")
    def inference(self, image_path):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
        }

        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": "What’s in this image?"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        captions = response.json()["choices"][0]["message"]["content"]
        print(f"\nProcessed ImageCaptioning, Input Image: {image_path}, Output Text: {captions}")
        return captions


class Segmenting:
    def __init__(self, device):
        print(f"Inintializing Segmentation to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_checkpoint_path = os.path.join("checkpoints","sam")

        self.download_parameters()
        self.sam = build_sam(checkpoint=self.model_checkpoint_path).to(device)
        self.sam_predictor = SamPredictor(self.sam)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        
        self.saved_points = []
        self.saved_labels = []

    def download_parameters(self):
        url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        if not os.path.exists(self.model_checkpoint_path):
            wget.download(url,out=self.model_checkpoint_path)

        
    def show_mask(self, mask: np.ndarray,image: np.ndarray,
                random_color: bool = False, transparency=1) -> np.ndarray:
        
        """Visualize a mask on top of an image.
        Args:
            mask (np.ndarray): A 2D array of shape (H, W).
            image (np.ndarray): A 3D array of shape (H, W, 3).
            random_color (bool): Whether to use a random color for the mask.
        Outputs:
            np.ndarray: A 3D array of shape (H, W, 3) with the mask
            visualized on top of the image.
            transparenccy: the transparency of the segmentation mask
        """
        
        if random_color:
            color = np.concatenate([np.random.random(3)], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) * 255

        image = cv2.addWeighted(image, 0.7, mask_image.astype('uint8'), transparency, 0)


        return image

    def show_box(self, box, ax, label):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
        ax.text(x0, y0, label)

    
    def get_mask_with_boxes(self, image_pil, image, boxes_filt):

        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(self.device)

        masks, _, _ = self.sam_predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes.to(self.device),
            multimask_output = False,
        )
        return masks
    
    def segment_image_with_boxes(self, image_pil, image_path, boxes_filt, pred_phrases):

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam_predictor.set_image(image)

        masks = self.get_mask_with_boxes(image_pil, image, boxes_filt)

        # draw output image

        for mask in masks:
            image = self.show_mask(mask[0].cpu().numpy(), image, random_color=True, transparency=0.3)

        updated_image_path = get_new_image_name(image_path, func_name="segmentation")
        
        new_image = Image.fromarray(image)
        new_image.save(updated_image_path)

        return updated_image_path

    def set_image(self, img) -> None:
        """Set the image for the predictor."""
        with torch.cuda.amp.autocast():
            self.sam_predictor.set_image(img)

    def show_points(self, coords: np.ndarray, labels: np.ndarray,
                image: np.ndarray) -> np.ndarray:
        """Visualize points on top of an image.

        Args:
            coords (np.ndarray): A 2D array of shape (N, 2).
            labels (np.ndarray): A 1D array of shape (N,).
            image (np.ndarray): A 3D array of shape (H, W, 3).
        Returns:
            np.ndarray: A 3D array of shape (H, W, 3) with the points
            visualized on top of the image.
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        for p in pos_points:
            image = cv2.circle(
                image, p.astype(int), radius=3, color=(0, 255, 0), thickness=-1)
        for p in neg_points:
            image = cv2.circle(
                image, p.astype(int), radius=3, color=(255, 0, 0), thickness=-1)
        return image


    def segment_image_with_click(self, img, is_positive: bool,
                            evt: gr.SelectData):
                            
        self.sam_predictor.set_image(img)
        self.saved_points.append([evt.index[0], evt.index[1]])
        self.saved_labels.append(1 if is_positive else 0)
        input_point = np.array(self.saved_points)
        input_label = np.array(self.saved_labels)
        
        # Predict the mask
        with torch.cuda.amp.autocast():
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

        img = self.show_mask(masks[0], img, random_color=False, transparency=0.3)

        img = self.show_points(input_point, input_label, img)

        return img

    def segment_image_with_coordinate(self, img, is_positive: bool,
                            coordinate: tuple):
        '''
            Args:
                img (numpy.ndarray): the given image, shape: H x W x 3.
                is_positive: whether the click is positive, if want to add mask use True else False.
                coordinate: the position of the click
                          If the position is (x,y), means click at the x-th column and y-th row of the pixel matrix.
                          So x correspond to W, and y correspond to H.
            Output:
                img (PLI.Image.Image): the result image
                result_mask (numpy.ndarray): the result mask, shape: H x W

            Other parameters:
                transparency (float): the transparenccy of the mask
                                      to control he degree of transparency after the mask is superimposed.
                                      if transparency=1, then the masked part will be completely replaced with other colors.
        '''
        self.sam_predictor.set_image(img)
        self.saved_points.append([coordinate[0], coordinate[1]])
        self.saved_labels.append(1 if is_positive else 0)
        input_point = np.array(self.saved_points)
        input_label = np.array(self.saved_labels)

        # Predict the mask
        with torch.cuda.amp.autocast():
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )


        img = self.show_mask(masks[0], img, random_color=False, transparency=0.3)

        img = self.show_points(input_point, input_label, img)

        img = Image.fromarray(img)
        
        result_mask = masks[0]

        return img, result_mask

    @prompts(name="Segment the Image",
             description="useful when you want to segment all the part of the image, but not segment a certain object."
                         "like: segment all the object in this image, or generate segmentations on this image, "
                         "or segment the image,"
                         "or perform segmentation on this image, "
                         "or segment all the object in this image."
                         "The input to this tool should be a string, representing the image_path")
    def inference_all(self,image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image)
        plt.figure(figsize=(20,20))
        plt.imshow(image)
        if len(masks) == 0:
            return
        sorted_anns = sorted(masks, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        color = []
        for ann in sorted_anns:
            m = ann['segmentation']
            img = np.ones((m.shape[0], m.shape[1], 3))
            color_mask = np.random.random((1, 3)).tolist()[0]
            for i in range(3):
                img[:,:,i] = color_mask[i]
            ax.imshow(np.dstack((img, m)))

        updated_image_path = get_new_image_name(image_path, func_name="segment-image")
        plt.axis('off')
        plt.savefig(
            updated_image_path, 
            bbox_inches="tight", dpi=300, pad_inches=0.0
        )
        return updated_image_path


class Text2Box:
    def __init__(self, device):
        print(f"Initializing ObjectDetection to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_checkpoint_path = os.path.join("checkpoints","groundingdino")
        self.model_config_path = os.path.join("checkpoints","grounding_config.py")
        self.download_parameters()
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.grounding = (self.load_model()).to(self.device)

    def download_parameters(self):
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        if not os.path.exists(self.model_checkpoint_path):
            wget.download(url,out=self.model_checkpoint_path)
        config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        if not os.path.exists(self.model_config_path):
            wget.download(config_url,out=self.model_config_path)
    def load_image(self,image_path):
         # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([512], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def load_model(self):
        args = SLConfig.fromfile(self.model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def get_grounding_boxes(self, image, caption, with_logits=True):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.grounding(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = self.grounding.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases
    
    def plot_boxes_to_image(self, image_pil, tgt):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        assert len(boxes) == len(labels), "boxes and labels must have same length"

        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        # draw boxes and masks
        for box, label in zip(boxes, labels):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            # draw.text((x0, y0), str(label), fill=color)

            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)
            # bbox = draw.textbbox((x0, y0), str(label))
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white")

            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=2)

        return image_pil, mask
    
    @prompts(name="Detect the Give Object",
             description="useful when you only want to detect or find out given objects in the picture"  
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path, the text description of the object to be found")
    def inference(self, inputs):
        image_path, det_prompt = inputs.split(",")
        print(f"image_path={image_path}, text_prompt={det_prompt}")
        image_pil, image = self.load_image(image_path)

        boxes_filt, pred_phrases = self.get_grounding_boxes(image, det_prompt)

        size = image_pil.size
        pred_dict = {
        "boxes": boxes_filt,
        "size": [size[1], size[0]],  # H,W
        "labels": pred_phrases,}

        image_with_box = self.plot_boxes_to_image(image_pil, pred_dict)[0]

        updated_image_path = get_new_image_name(image_path, func_name="detect-something")
        updated_image = image_with_box.resize(size)
        updated_image.save(updated_image_path)
        print(
            f"\nProcessed ObejectDetecting, Input Image: {image_path}, Object to be Detect {det_prompt}, "
            f"Output Image: {updated_image_path}")
        return updated_image_path


class Inpainting:
    def __init__(self, device):
        self.device = device

    def __call__(self, prompt, image, mask_image):
        client = dalle(api_key=API_KEY)
        response = client.images.edit(
        model="dall-e-2",
        image=open(image, "rb"),
        mask=open(mask_image, "rb"),
        prompt=prompt,
        n=1,
        size="1024x1024"
        )
        image_url = response.data[0].url

        response = requests.get(image_url)
        update_image = Image.open(BytesIO(response.content))
        return update_image


class ImageEditing:
    template_model = True
    def __init__(self, Text2Box:Text2Box, Segmenting:Segmenting, Inpainting:Inpainting):
        print(f"Initializing ImageEditing")
        self.sam = Segmenting
        self.grounding = Text2Box
        self.inpaint = Inpainting

    def pad_edge(self,mask,padding):
        #mask Tensor [H,W]
        mask = mask.numpy()
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
            mask_array[padded_slice] = True
        new_mask = (mask_array * 255).astype(np.uint8)
        #new_mask
        return new_mask

    @prompts(name="Remove Something From The Photo",
             description="useful when you want to remove and object or something from the photo "
                         "from its description or location. "
                         "The input to this tool should be a comma separated string of two, "
                         "representing the image_path and the object need to be removed. ")    
    def inference_remove(self, inputs):
        image_path, to_be_removed_txt = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        return self.inference_replace_sam(f"{image_path},{to_be_removed_txt},background")

    @prompts(name="Replace Something From The Photo",
            description="useful when you want to replace an object from the object description or "
                        "location with another object from its description. "
                        "The input to this tool should be a comma separated string of three, "
                        "representing the image_path, the object to be replaced, the object to be replaced with ")
    def inference_replace_sam(self,inputs):
        image_path, to_be_replaced_txt, replace_with_txt = inputs.split(",")
        
        print(f"image_path={image_path}, to_be_replaced_txt={to_be_replaced_txt}")
        image_pil, image = self.grounding.load_image(image_path)
        boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(image, to_be_replaced_txt)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.sam.sam_predictor.set_image(image)
        masks = self.sam.get_mask_with_boxes(image_pil, image, boxes_filt)
        mask = torch.sum(masks, dim=0).unsqueeze(0)
        mask = torch.where(mask > 0, True, False)
        mask = mask.squeeze(0).squeeze(0).cpu() #tensor

        mask = self.pad_edge(mask,padding=20) #numpy
        mask_image = Image.fromarray(mask)

        image_filename_image_pil = os.path.join('PIL', f"{str(uuid.uuid4())[:8]}.png")
        image_pil.save(image_filename_image_pil)
        image_filename_mask_image = os.path.join('PIL', f"{str(uuid.uuid4())[:8]}.png")
        mask_image.save(image_filename_mask_image)

        updated_image = self.inpaint(prompt=replace_with_txt, image=image_filename_image_pil,
                                     mask_image=image_filename_mask_image)
        updated_image_path = get_new_image_name(image_path, func_name="replace-something")
        updated_image = updated_image.resize(image_pil.size)
        updated_image.save(updated_image_path)
        print(
            f"\nProcessed ImageEditing, Input Image: {image_path}, Replace {to_be_replaced_txt} to {replace_with_txt}, "
            f"Output Image: {updated_image_path}")
        return updated_image_path


class Imagic:
    def __init__(self, device):
        print(f"Initializing Imagic to {device}")
        self.device = device
    
    @prompts(name="Imagic",
             description="useful when you want to change the pose of the living things in the photo. "
                         "like: make the dog sit, or make the cat jump. "
                         "The input to this tool should be a comma separated string of two. "
                         "representing the image_path and the text. ")

    def inference(self, inputs):
        image_path, text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        updated_image_path = get_new_image_name(image_path, func_name="imagic")

        print("===>Starting Imagic Training")
        command = [
            "accelerate", "launch", "train_imagic.py",
            "--pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4",
            "--output_dir=stable_diffusion_weights/imagic",
            "--input_image=" + image_path,
            "--target_text=" + text,
            "--seed=3434554",
            "--resolution=256",
            "--mixed_precision=fp16",
            "--use_8bit_adam",
            "--gradient_accumulation_steps=1",
            "--emb_learning_rate=1e-3",
            "--learning_rate=1e-6",
            "--emb_train_steps=500",
            "--max_train_steps=1000",
            "--gradient_checkpointing"
        ]

        subprocess.run(command)


        print("===>Starting Imagic Inference")

        model_path = "stable_diffusion_weights/imagic"

        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        pipe = StableDiffusionPipeline.from_pretrained(model_path, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
        target_embeddings = torch.load(os.path.join(model_path, "target_embeddings.pt")).to("cuda")
        optimized_embeddings = torch.load(os.path.join(model_path, "optimized_embeddings.pt")).to("cuda")
        g_cuda = torch.Generator(device="cuda")
        seed = 1234
        g_cuda.manual_seed(seed)

        alpha = 0.9
        num_samples = 4
        guidance_scale = 3
        num_inference_steps = 50
        height = 256
        width = 256

        edit_embeddings = alpha*target_embeddings + (1-alpha)*optimized_embeddings

        with autocast("cuda"), torch.inference_mode():
            images = pipe(
                prompt_embeds=edit_embeddings,
                height=height,
                width=width,
                num_images_per_prompt=num_samples,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=g_cuda
            ).images

        for img in images:
            img.save(updated_image_path)
        
        print(f"\nProcessed Imagic, Input Image: {image_path}, Instruct Text: {text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path


class ConversationBot:
    def __init__(self, load_dict):
        # load_dict = {'VisualQuestionAnswering':'cuda:0', 'ImageCaptioning':'cuda:1',...}
        print(f"Initializing VisualChatGPT, load_dict={load_dict}")
        if 'ImageCaptioning' not in load_dict:
            raise ValueError("You have to load ImageCaptioning as a basic function for VisualChatGPT")

        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k!='self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})
        
        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.llm = ChatOpenAI(openai_api_key=API_KEY, model_name="gpt-4-1106-preview", temperature=0)
        self.memory = ConversationTokenBufferMemory(llm=self.llm, memory_key="chat_history", output_key='output', max_token_limit=500)

    def init_agent(self, lang):
        self.memory.clear() #clear previous history
        if lang=='English':
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = VISUAL_CHATGPT_PREFIX, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS, VISUAL_CHATGPT_SUFFIX
            place = "Enter text and press enter, or upload an image"
            label_clear = "Clear"
        else:
            PREFIX, FORMAT_INSTRUCTIONS, SUFFIX = VISUAL_CHATGPT_PREFIX_CN, VISUAL_CHATGPT_FORMAT_INSTRUCTIONS_CN, VISUAL_CHATGPT_SUFFIX_CN
            place = "输入文字并回车，或者上传图片"
            label_clear = "清除"
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX}, )
        return gr.update(visible = True), gr.update(visible = False), gr.update(placeholder=place), gr.update(value=label_clear)

    def run_text(self, text, state):
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.load_memory_variables({})}")
        return state, state

    def run_image(self, image, state, txt, lang):
        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        print("======>Auto Resize Image...")
        img = Image.open(image.name)
        width, height = img.size
        ratio = min(512 / width, 512 / height)
        width_new, height_new = (round(width * ratio), round(height * ratio))
        width_new = int(np.round(width_new / 64.0)) * 64
        height_new = int(np.round(height_new / 64.0)) * 64
        img = img.resize((width_new, height_new))
        img = img.convert('RGB')
        img.save(image_filename, "PNG")
        print(f"Resize image form {width}x{height} to {width_new}x{height_new}")
        description = self.models['ImageCaptioning'].inference(image_filename)
        if lang == 'Chinese':
            Human_prompt = f'\nHuman: 提供一张名为 {image_filename}的图片。它的描述是: {description}。 这些信息帮助你理解这个图像，但是你应该使用工具来完成下面的任务，而不是直接从我的描述中想象。 如果你明白了, 说 \"收到\". \n'
            AI_prompt = "收到。  "
        else:
            Human_prompt = f'\nHuman: provide a figure named {image_filename}. The description is: {description}. This information helps you to understand this image, but you should use tools to finish following tasks, rather than directly imagine from my description. If you understand, say \"Received\". \n'
            AI_prompt = "Received.  "
        self.agent.memory.save_context({"input": Human_prompt}, {"output": AI_prompt})
        state = state + [(f"![](file={image_filename})*{image_filename}*", AI_prompt)]
        print(f"\nProcessed run_image, Input image: {image_filename}\nCurrent state: {state}\n"
              f"Current Memory: {self.agent.memory.load_memory_variables({})}")
        return state, state, f'{txt} {image_filename} '


if __name__ == '__main__':
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default="ImageCaptioning_cuda:0,Text2Image_cuda:0")
    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}
    bot = ConversationBot(load_dict=load_dict)
    with gr.Blocks(css="#chatbot .overflow-y-auto{height:500px}") as demo:
        lang = gr.Radio(choices = ['Chinese','English'], value=None, label='Language')
        chatbot = gr.Chatbot(elem_id="chatbot", label="Visual ChatGPT")
        state = gr.State([])
        with gr.Row(visible=False) as input_raws:
            with gr.Column(scale=0.7):
                txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter, or upload an image").style(
                    container=False)
            with gr.Column(scale=0.15, min_width=0):
                clear = gr.Button("Clear")
            with gr.Column(scale=0.15, min_width=0):
                btn = gr.UploadButton(label="🖼️",file_types=["image"])

        lang.change(bot.init_agent, [lang], [input_raws, lang, txt, clear])
        txt.submit(bot.run_text, [txt, state], [chatbot, state])
        txt.submit(lambda: "", None, txt)
        btn.upload(bot.run_image, [btn, state, txt, lang], [chatbot, state, txt])
        clear.click(bot.memory.clear)
        clear.click(lambda: [], None, chatbot)
        clear.click(lambda: [], None, state)
    demo.queue()
    demo.launch(share=True)
