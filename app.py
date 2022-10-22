import io
import base64
import os
import sys
from styleGan import *
import numpy as np
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline
from PIL import Image
from PIL import ImageOps
import gradio as gr
import base64
import skimage
import skimage.measure
from utils import *
import random
import webbrowser
import calendar
import time
from shutil import rmtree


global log_name, time_stamp 
current_GMT = time.gmtime()
time_stamp=str(calendar.timegm(current_GMT))
log_name = "log/"+ time_stamp+".txt"



with open(log_name,"w") as f:
    f.write("logNo: "+time_stamp)

if os.path.exists("log"):
    rmtree("log")
os.mkdir("log")


sys.path.append("./glid_3_xl_stable")

USE_GLID = True
try:
    from glid3xlmodel import GlidModel
except:
    USE_GLID = False

try:
    cuda_available = torch.cuda.is_available()
except:
    cuda_available = False
finally:
    if sys.platform == "darwin":
        device = "mps"
    elif cuda_available:
        device = "cuda"
    else:
        device = "cpu"

if device != "cuda":
    import contextlib

    autocast = contextlib.nullcontext


def load_html():
    body, canvaspy = "", ""
    with open("index.html", encoding="utf8") as f:
        body = f.read()
    with open("canvas.py", encoding="utf8") as f:
        canvaspy = f.read()
    body = body.replace("- paths:\n", "")
    body = body.replace("  - ./canvas.py\n", "")
    body = body.replace("from canvas import InfCanvas", canvaspy)
    return body


def test(x):
    x = load_html()
    return f"""<iframe id="sdinfframe" style="width: 100%; height: 700px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{x}'></iframe>"""


DEBUG_MODE = False

try:
    SAMPLING_MODE = Image.Resampling.LANCZOS
except Exception as e:
    SAMPLING_MODE = Image.LANCZOS

try:
    contain_func = ImageOps.contain
except Exception as e:

    def contain_func(image, size, method=SAMPLING_MODE):
        # from PIL: https://pillow.readthedocs.io/en/stable/reference/ImageOps.html#PIL.ImageOps.contain
        im_ratio = image.width / image.height
        dest_ratio = size[0] / size[1]
        if im_ratio != dest_ratio:
            if im_ratio > dest_ratio:
                new_height = int(image.height / image.width * size[0])
                if new_height != size[1]:
                    size = (size[0], new_height)
            else:
                new_width = int(image.width / image.height * size[1])
                if new_width != size[0]:
                    size = (new_width, size[1])
        return image.resize(size, resample=method)


PAINT_SELECTION = "✥"
IMAGE_SELECTION = "🖼️"
BRUSH_SELECTION = "🖌️"
blocks = gr.Blocks()
model = {}


def get_token():
    token = "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"
    if os.path.exists(".token"):
        with open(".token", "r") as f:
            token = f.read()
    token = os.environ.get("hftoken", token)
    return token


def save_token(token):
    with open(".token", "w") as f:
        f.write(token)


class StableDiffusion:
    def __init__(self, token=""):
        self.token = token
        if device == "cuda":
            text2img = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                revision="fp16",
                torch_dtype=torch.float16,
                use_auth_token=token,
            ).to(device)
        else:
            text2img = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", use_auth_token=token,
            ).to(device)
        if device == "mps":
            _ = text2img("", num_inference_steps=1)
        self.safety_checker = text2img.safety_checker
        inpaint = StableDiffusionInpaintPipeline(
            vae=text2img.vae,
            text_encoder=text2img.text_encoder,
            tokenizer=text2img.tokenizer,
            unet=text2img.unet,
            scheduler=text2img.scheduler,
            safety_checker=text2img.safety_checker,
            feature_extractor=text2img.feature_extractor,
        ).to(device)
        save_token(token)
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory // (
                1024 ** 3
            )
            if total_memory <= 5:
                inpaint.enable_attention_slicing()
        except:
            pass
        self.text2img = text2img
        self.inpaint = inpaint

    def run(
        self,
        image_pil,
        prompt="",
        guidance_scale=7.5,
        resize_check=True,
        enable_safety=True,
        fill_mode="patchmatch",
        strength=0.75,
        step=50,
        **kwargs,
    ):
        text2img, inpaint = self.text2img, self.inpaint
        if enable_safety:
            text2img.safety_checker = self.safety_checker
            inpaint.safety_checker = self.safety_checker
        else:
            text2img.safety_checker = lambda images, **kwargs: (images, False)
            inpaint.safety_checker = lambda images, **kwargs: (images, False)

        sel_buffer = np.array(image_pil)
        img = sel_buffer[:, :, 0:3]
        mask = sel_buffer[:, :, -1]
        process_size = 512 if resize_check else model["sel_size"]
        if mask.sum() > 0:
            img, mask = functbl[fill_mode](img, mask)
            init_image = Image.fromarray(img)
            mask = 255 - mask
            mask = skimage.measure.block_reduce(mask, (8, 8), np.max)
            mask = mask.repeat(8, axis=0).repeat(8, axis=1)
            mask_image = Image.fromarray(mask)
            # mask_image=mask_image.filter(ImageFilter.GaussianBlur(radius = 8))
            with autocast("cuda"):
                images = inpaint(
                    prompt=prompt,
                    init_image=init_image.resize(
                        (process_size, process_size), resample=SAMPLING_MODE
                    ),
                    mask_image=mask_image.resize((process_size, process_size)),
                    strength=strength,
                    num_inference_steps=step,
                    guidance_scale=guidance_scale,
                )["sample"]
        else:
            with autocast("cuda"):
                images = text2img(
                    prompt=prompt, height=process_size, width=process_size,
                )["sample"]
        return images[0]


def get_model(token="", model_choice=""):
    if "model" not in model:
        if not USE_GLID:
            model_choice = "stablediffusion"
        if model_choice == "stablediffusion":
            tmp = StableDiffusion(token)
        else:
            config_lst = ["--edit", "a.png", "--mask", "mask.png"]
            if device == "cpu":
                config_lst.append("--cpu")
            tmp = GlidModel(config_lst)
        model["model"] = tmp
    return model["model"]


def run_outpaint(
    sel_buffer_str,
    prompt_text,
    strength,
    #guidance,
    #step,
    #resize_check,
    fill_mode,
    #enable_safety,
    state,
):
    base64_str = "base64"
    data = base64.b64decode(str(sel_buffer_str))
    pil = Image.open(io.BytesIO(data))
    sel_buffer = np.array(pil)
    cur_model = get_model()
    image = cur_model.run(
        image_pil=pil,
        prompt=prompt_text,
        guidance_scale=sd_guidance,
        strength=strength,
        step=sd_step,
        resize_check=sd_resize,
        fill_mode=fill_mode,
        enable_safety=safety_check,
        width=max(model["sel_size"], 512),
        height=max(model["sel_size"], 512),
    )
    out = sel_buffer.copy()
    out[:, :, 0:3] = np.array(
        image.resize((model["sel_size"], model["sel_size"]), resample=SAMPLING_MODE,)
    )
    out[:, :, -1] = 255
    out_pil = Image.fromarray(out)
    out_buffer = io.BytesIO()
    out_pil.save(out_buffer, format="PNG")
    out_buffer.seek(0)
    base64_bytes = base64.b64encode(out_buffer.read())
    base64_str = base64_bytes.decode("ascii")
    imgLog(out_pil,prompt_text,fill_mode,strength)
    return (
        gr.update(label=str(state + 1), value=base64_str,),
        gr.update(label="Prompt"),
        state + 1,
    )
def openPage(x):
    return webbrowser.open_new_tab("baristerzi.com")

def load_js(name):
    if name in ["export", "commit", "undo"]:
        return f"""
function (x)
{{ 
    let frame=document.querySelector("gradio-app").shadowRoot.querySelector("#sdinfframe").contentWindow.document;
    let button=frame.querySelector("#{name}");
    button.click();
    return x;
}}
"""
    ret = ""
    with open(f"./js/{name}.js", "r") as f:
        ret = f.read()
    return ret

def load_js1(name1,name2):
    if name1 in ["export", "commit", "undo"]:
        return f"""
function (x)
{{ 
    let frame=document.querySelector("gradio-app").shadowRoot.querySelector("#sdinfframe").contentWindow.document;
    let button=frame.querySelector("#{name}");
    button.click();
    return x;
}}
"""
    ret1 = ""
    ret2 = ""
    with open(f"./js/{name1}.js", "r") as f:
        ret1 = f.read()
    with open(f"./js/{name2}.js", "r") as f:
        ret2 = f.read()
    return ret1,ret2


def imgLogS(img,fs,fc,fac):
    current_GMT = time.gmtime()
    imgName=str(time_stamp)+"_"+str(calendar.timegm(current_GMT))+".png"
    img.save("log/"+imgName)
    with open(log_name,"a") as f:
        f.write("\n")
        f.write("___styleGan:")
        f.write("\n")
        f.write("imgName: "+imgName)
        f.write("\n")
        f.write("sourceSeed: "+str(fs))
        f.write("\n")
        f.write("targetSeed: "+str(fc))
        f.write("\n")
        f.write("factor: "+str(fac))
        f.write("\n")
        f.write("___")

def imgLogA(img):
    current_GMT = time.gmtime()
    imgName=str(time_stamp)+"_"+str(calendar.timegm(current_GMT))+".png"
    img.save("log/"+imgName)
    with open(log_name,"a") as f:
        f.write("\n")
        f.write("___Apply Image:")
        f.write("\n")
        f.write("imgName: "+imgName)
        f.write("\n")
        f.write("___")

def imgLog(img,promt,initM,strength):
    current_GMT = time.gmtime()
    imgName=str(time_stamp)+"_"+str(calendar.timegm(current_GMT))+".png"
    img.save("log/"+imgName)
    with open(log_name,"a") as f:
        f.write("\n")
        f.write("___stableDiffusion:")
        f.write("\n")
        f.write("promt : "+promt)
        f.write("\n")
        f.write("init_mode : "+initM)
        f.write("\n")
        f.write("strength : "+str(strength))
        f.write("\n")
        f.write("imgName: "+imgName)
        f.write("\n")
        f.write("___")

    
#addition func
def dnm(fImg,sImg,fac):
    fac=fac*100

    rgb_img=fImg.convert("RGB")
    r,g,b=rgb_img.getpixel((0, 0))
    fc=int(str(r)+str(g)+str(b))
    rgb_img=sImg.convert("RGB")
    r,g,b=rgb_img.getpixel((0, 0))
    sc=int(str(r)+str(g)+str(b))
    img=makeInt(fc,sc,fac,G,device)
    imgLogS(img,fc,sc,fac)
    return img


def rnd(fac):
    fac=fac*100
    s1=random.randint(0,1000)
    s2=random.randint(0,1000)
    img=makeInt(s1,s2,fac,G,device)
    imgLogS(img,s1,s2,fac)
    return img


#addition list
from glob import glob
exa=glob("syntheticSections/*")
exa=random.sample(exa,100)
len(exa)




upload_button_js = load_js("upload")
outpaint_button_js = load_js("outpaint")
proceed_button_js = load_js("proceed")
mode_js = load_js("mode")
setup_button_js = load_js("setup1")
#setup_button_js = load_js1("setup","upload")


safety_check =True
sd_guidance=7.5
sd_step=50
sd_resize=True

token=get_token()
with blocks as demo:
    # title
    title = gr.Markdown(
        """
    **SectionTool**: Section tool for auditorium  early design stage.
    User guide [video](https://github.com/lkwq007/stablediffusion-infinity), [pdf](https://github.com/lkwq007/stablediffusion-infinity)
    """
    )
   
    
    with gr.Row():
        with gr.Accordion("Section library"):
            with gr.Row():
                gr.Gallery(exa,label="Click to enlarge").style(grid=10)
    with gr.Row():
        gr.Markdown("Select a section from the gallery and drop it into the image boxes below.")
    with gr.Row():
        with gr.Column(scale=2,min_width=250):
            title=gr.Markdown("Upload source section:")
            sourceImg=gr.Image(image_mode="RGBA", source="upload", type="pil", interactive=True)
            #conf=sourceImg.get_config()
        with gr.Column(scale=1,min_width=150):
            title=gr.Markdown("Change factor and generate:")
            fac=gr.Slider(0,1,step=0.05, value=0.5, label="Factor:")
            genBut=gr.Button(value="Generate Section")
            title=gr.Markdown("or generate random section:")
            randBut=gr.Button(value="Random Section")
        with gr.Column(scale=2,min_width=250):
            title=gr.Markdown("Upload target section:")
            targetImg=gr.Image(image_mode="RGBA", source="upload", type="pil")
        with gr.Column(scale=2,min_width=250):
            title=gr.Markdown("New section:")
            genImg=gr.Image(image_mode="RGBA", source="upload", type="pil", interactive=True)
            upload_button=gr.Button(value="Apply section")
    

    inList=[sourceImg,targetImg,fac]
    
    genBut.click(dnm,inputs=inList,outputs=genImg)
    randBut.click(rnd,fac,outputs=genImg)
    
    # frame
    with gr.Row():
        frame = gr.HTML(test(2), visible=True)

    # setup
    """with gr.Row():
        with gr.Column(scale=4, min_width=350):
            token = gr.Textbox(
                label="Huggingface token",
                value=get_token(),
                placeholder="Input your token here",
            )
        with gr.Column(scale=3, min_width=320):
            model_selection = gr.Radio(
                label="Model",
                choices=["stablediffusion", "glid-3-xl-stable"],
                value="stablediffusion",
            )
        with gr.Column(scale=1, min_width=100):
            canvas_width = gr.Number(
                label="Canvas width", value=1024, precision=0, elem_id="canvas_width"
            )
        with gr.Column(scale=1, min_width=100):
            canvas_height = gr.Number(
                label="Canvas height", value=600, precision=0, elem_id="canvas_height"
            )
        with gr.Column(scale=1, min_width=100):
            selection_size = gr.Number(
                label="Selection box size",
                value=256,
                precision=0,
                elem_id="selection_size",
            )
    setup_button = gr.Button("Setup (may take a while)", variant="primary")"""

    
    with gr.Row():
        with gr.Column(scale=6, min_width=120,elem_id="control"):
            
            control_text=gr.Markdown("**Control**")
            # canvas control
            with gr.Row():
                canvas_control = gr.Radio(label="",
                    choices=[PAINT_SELECTION, IMAGE_SELECTION, BRUSH_SELECTION],
                    value=PAINT_SELECTION,
                    #elem_id="control",
                )
            with gr.Row():
                with gr.Column(scale=2, min_width=48):
                    run_button = gr.Button(value="Outpaint")
                with gr.Column(scale=1, min_width=24):   
                    commit_button = gr.Button(value="✓")
                with gr.Column(scale=1, min_width=24):
                    retry_button = gr.Button(value="⟳")
                with gr.Column(scale=1, min_width=24):
                    undo_button = gr.Button(value="↶")
            


            
        with gr.Column(scale=11, min_width=220):
            sd_prompt = gr.Textbox(
                label="Prompt", placeholder="input your prompt here", lines=5
            )
        with gr.Column(scale=16, min_width=320):
            """with gr.Box():
                sd_resize = gr.Checkbox(label="Resize input to 515x512", value=True)
                safety_check = gr.Checkbox(label="Enable Safety Checker", value=True)"""
            
            with gr.Row():
                init_mode = gr.Radio(
                    label="Init mode",
                    choices=[
                        "patchmatch",
                        "edge_pad",
                        "cv2_ns",
                        "cv2_telea",
                        "gaussian",
                        "perlin",
                    ],
                    value="patchmatch",
                    type="value",
                )
            sd_strength = gr.Slider(
                label="Strength", minimum=0.0, maximum=1.0, value=0.75, step=0.01)
            
        """with gr.Column(scale=1, min_width=150):
            sd_step = gr.Number(label="Step", value=50, precision=0)
            sd_guidance = gr.Number(label="Guidance", value=7.5)"""
    with gr.Row():
        export_button = gr.Button(value="Save image")
    with gr.Row():
        exText=gr.Markdown("Save the image and fill out the evaluation form(Turkish). [Evaluation](https://baristerzi.com) form if not opened.")

    """with gr.Row():
        with gr.Column(scale=4, min_width=600):
            init_mode = gr.Radio(
                label="Init mode",
                choices=[
                    "patchmatch",
                    "edge_pad",
                    "cv2_ns",
                    "cv2_telea",
                    "gaussian",
                    "perlin",
                ],
                value="patchmatch",
                type="value",
            )"""
    with gr.Row():
        refTitle = gr.Markdown(
        """
        ***SectionTool***: Section tool for auditorium  early design stage: [https://github.com/baristerzi/SectionTool](https://github.com/baristerzi/SectionTool) \n
        **references:** \n
        *StyleGAN2-ADA*: StyleGAN2 with adaptive discriminator augmentation: [https://github.com/NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) \n
        *stablediffusion-infinity*: Outpainting with Stable Diffusion on an infinite canvas: [https://github.com/lkwq007/stablediffusion-infinity](https://github.com/lkwq007/stablediffusion-infinity)\n

        Created by [Barış Terzi](https://github.com/baristerzi), to contact: [baris@virtualogie.com](mailto:baris@virtualogie.com)
        """
        )
        

    proceed_button = gr.Button("Proceed", elem_id="proceed", visible=DEBUG_MODE)
    # sd pipeline parameters
    """with gr.Accordion("Upload image", open=False):
        image_box = gr.Image(image_mode="RGBA", source="upload", type="pil")
        upload_button = gr.Button(
            "Before uploading the image you need to setup the canvas first"
        )"""
    model_output = gr.Textbox(visible=DEBUG_MODE, elem_id="output", label="0")
    model_input = gr.Textbox(visible=DEBUG_MODE, elem_id="input", label="Input")
    upload_output = gr.Textbox(visible=DEBUG_MODE, elem_id="upload", label="0")
    model_output_state = gr.State(value=0)
    upload_output_state = gr.State(value=0)
    # canvas_state = gr.State({"width":1024,"height":600,"selection_size":384})

    def upload_func(image, state):
        pil = image.convert("RGBA")
        w, h = pil.size
        if w > model["width"] - 100 or h > model["height"] - 100:
            pil = contain_func(pil, (model["width"] - 100, model["height"] - 100))
        out_buffer = io.BytesIO()
        pil.save(out_buffer, format="PNG")
        out_buffer.seek(0)
        base64_bytes = base64.b64encode(out_buffer.read())
        base64_str = base64_bytes.decode("ascii")
        frame: gr.update(visible=True)
        imgLogA(image)

        return (
            gr.update(label=str(state + 1), value=base64_str),
            state + 1,
            
        ) #frame: gr.update(visible=True),

    upload_button.click(
        fn=upload_func,
        inputs=[genImg, upload_output_state],
        outputs=[upload_output, upload_output_state],
        _js=setup_button_js,
    )

    """setup_button.click(
        fn=setup_func,
        inputs=[token, canvas_width, canvas_height, selection_size, model_selection],
        outputs=[
            token,
            canvas_width,
            canvas_height,
            selection_size,
            setup_button,
            frame,
            upload_button,
            model_selection,
        ],
        _js=setup_button_js,
    )"""
    run_button.click(
        fn=None, inputs=[run_button], outputs=[run_button], _js=outpaint_button_js,
    )
    retry_button.click(
        fn=None, inputs=[run_button], outputs=[run_button], _js=outpaint_button_js,
    )
    proceed_button.click(
        fn=run_outpaint,
        inputs=[
            model_input,
            sd_prompt,
            sd_strength,
            #sd_guidance,
            #sd_step,
            #sd_resize,
            init_mode,
            #safety_check,
            model_output_state,
        ],
        outputs=[model_output, sd_prompt, model_output_state],
        _js=proceed_button_js,
    )
    export_button.click(
        fn=openPage, inputs=[export_button], outputs=[export_button], _js=load_js("export")
    )
    commit_button.click(
        fn=None, inputs=[export_button], outputs=[export_button], _js=load_js("commit")
    )
    undo_button.click(
        fn=None, inputs=[export_button], outputs=[export_button], _js=load_js("undo")
    )
    canvas_control.change(
        fn=None, inputs=[canvas_control], outputs=[canvas_control], _js=mode_js,
    )

def setup_func(token_val, width, height, size, model_choice):
        model["width"] = width
        model["height"] = height
        model["sel_size"] = size
        try:
            get_model(token_val, model_choice)
        except Exception as e:
            print(e)
            return {token: gr.update(value=str(e))}
        return {
            token: gr.update(visible=False),
            #canvas_width: gr.update(visible=False),
            #canvas_height: gr.update(visible=False),
            #selection_size: gr.update(visible=False),
            #setup_button: gr.update(visible=False),
            #frame: gr.update(visible=True),
            #upload_button: gr.update(value="Upload Image"),
            #model_selection: gr.update(visible=False),
        }

setup_func("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",1200,600,384,"stablediffusion")

PKL="../network-snapshot-000160.pkl"
print(f'Loading networks from "{PKL}"...')
device = torch.device('cuda')
with dnnlib.util.open_url(PKL) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="stablediffusion-infinity")
    parser.add_argument("--port", type=int, help="listen port", default=7860)
    parser.add_argument("--host", type=str, help="host", default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="share this app?")
    args = parser.parse_args()
    if args.share:
        try:
            import google.colab

            IN_COLAB = True
        except:
            IN_COLAB = False
        demo.launch(share=True, debug=IN_COLAB)
    else:
        demo.launch(server_name=args.host, server_port=args.port)

