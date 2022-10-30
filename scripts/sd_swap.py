import cv2, gc
import numpy as numpy
import mediapipe as mp
from resizeimage import resizeimage
import PIL
from PIL import Image, ImageDraw
import matplotlib.image
import os, argparse
import yaml, torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from tqdm import tqdm, trange
from einops import rearrange, repeat
from scipy.spatial import ConvexHull
import modules.scripts as scripts
import gradio as gr
from modules import processing, shared, sd_samplers, images, devices

from modules.processing import Processed
from modules.shared import opts, cmd_opts, state

def pil2numpy(image):
    image = numpy.asarray(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def numpy2pil(image):
    image = Image.fromarray(image)
    return image.convert("RGB")

class Script(scripts.Script):

    def title(self):
        return "Face Swap"


    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        return []


    def run(self, p):
        gc.collect()
        expand_up = 100 
        expand = 100
        all_faces = True
        write_pics_no_face_frames = False 
        init_image_pil = p.init_images[0]
        out_path = p.outpath_samples

        class MediaPipeFaceDetect:
            def __init__(self):
                mp_face_detection = mp.solutions.face_detection
                self.detector = mp_face_detection.FaceDetection(model_selection = 1, min_detection_confidence=0.2)

            def get_faces(self, frame):
                image_height, image_width, _ = frame.shape
                imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = self.detector.process(imgRGB)

                faces = []
                if results.detections: 
                    # Iterate over the found faces.
                    for face_no, face in enumerate(results.detections):			
                        face_bbox = face.location_data.relative_bounding_box
                        
                        x1 = int(face_bbox.xmin * image_width)
                        y1 = int(face_bbox.ymin * image_height)
                        w = int(face_bbox.width * image_width)
                        h = int(face_bbox.height * image_height)					

                        faces.append((x1, y1, w, h))
                print(f'got {len(faces)} faces')
                return faces

        num = 0

        init_image_np = pil2numpy(init_image_pil)
        faceDetect = MediaPipeFaceDetect()
        faces = faceDetect.get_faces(init_image_np)
        original_image_pil = init_image_pil.copy()
        p.batch_size = 1
        p.n_iter = 1

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode = True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        if len(faces) > 0:
            num += 1
            original_pil = original_image_pil

            for face in faces:		
                (x, y, w, h) = face
                x -= expand
                y -= expand_up
                w += expand * 2
                h += expand_up + expand
                box = (x, y, x + w, y + h)
                im_pil_cropped = original_pil.crop(box = box)
                im_pil_src = im_pil_cropped.copy()			
                (width, height) = im_pil_cropped.size
                if height > width:
                    new_width = 512
                    new_height = int((new_width / width) * height)
                else:
                    new_height = 512
                    new_width = int((new_height / height) * width)
                im_pil_preprocess = im_pil_cropped.resize((new_width, new_height))
                im_pil_preprocess = resizeimage.resize_cover(im_pil_preprocess, [512, 512])

                # do mesh image
                mesh_input = numpy.asarray(im_pil_preprocess)
                results = face_mesh.process(mesh_input)
                annotated_image_pil = im_pil_preprocess.copy()
                mask_pil = Image.new('RGB', (512, 512))
                if len(results.multi_face_landmarks) > 0:
                    pts = []
                    landmarks = results.multi_face_landmarks[0]
                    for landmark in landmarks.landmark:
                        pts.append((landmark.x, landmark.y))
                    hull = ConvexHull(pts)
                    verts = []
                    hull_vertices = [tuple(pts[i]) for i in hull.vertices]
                    for vertex in hull_vertices:
                        (vx, vy) = vertex
                        verts.append((vx * 512, vy * 512))
                    draw = ImageDraw.Draw(mask_pil)
                    draw.polygon(verts, fill=(256, 256, 256))
                    p.image_mask = mask_pil

                p.init_images = [im_pil_preprocess]
                processed = processing.process_images(p)
                output_image_pil = processed.images[0]
                if h > w:
                    new_h = h
                    new_w = int((new_h / 512) * 512)
                else:
                    new_w = w
                    new_h = int((new_w / 512) * 512)
                out_pil_rs_w = output_image_pil.resize((new_w, new_h))
                output_pil_back = resizeimage.resize_crop(out_pil_rs_w, [w, h])
                original_pil.paste(output_pil_back, (int(x), int(y)))
        else:
            original_pil = original_image_pil

        return Processed(p, [original_pil])