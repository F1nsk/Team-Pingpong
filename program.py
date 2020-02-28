#!/usr/bin/env python3

# Copyright (c) 2018 Anki, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License in the file LICENSE.txt or at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hello World

Make Vector say 'Hello World' in this simple Vector SDK example program.
"""
import threading
from queue import Queue

import anki_vector
from anki_vector.util import degrees, distance_mm, speed_mmps
from anki_vector.connection import ControlPriorityLevel
from anki_vector import events

import cv2
from pytracking.run_webcam import RobotTracker
import torchvision

import numpy as np

def get_encoded_labels():
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
        'N/A', 'N/A',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket',
        'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A',
        'dining table',
        'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    encoded_labels = {}
    for num, l in enumerate(COCO_INSTANCE_CATEGORY_NAMES):
        encoded_labels[num] = l

    return encoded_labels


counter = [0]


def on_new_raw_camera_image(robot, event_type, event, done, model,
                            output_queue, tracker):

    image = event.image
    image_tensor = torchvision.transforms.ToTensor()(image).cuda()

    if counter[0] % 10 == 0:
        print('DETECTION')
        outputs = model([image_tensor])[0]
        scores = outputs["scores"]
        boxes = outputs["boxes"][scores > 0.5]
        labels = outputs["labels"][scores > 0.5]
        numpy_labels = labels.detach().cpu().numpy()
        numpy_boxes = boxes.detach().cpu().numpy()
    else:
        numpy_boxes  = []
        numpy_labels = []

    counter[0]+=1

    if not tracker.initialized:
        for k, label in enumerate(numpy_labels):
            if label == 1:
                print('Initializing')
                tracker.init(np.asarray(image), numpy_boxes[k])
                print('initialized-....')
                exit(-1)
                break

    if tracker.initialized:
        tracked_box = tracker.track(np.asarray(image))
    else:
        tracked_box = None

    output_queue.put(dict(boxes=numpy_boxes,
                          labels=numpy_labels,
                          image=image,
                          tracked_box=tracked_box))
    print('===================================0')
    done.set()


def main():
        args = anki_vector.util.parse_command_args()
        encoded_labels = get_encoded_labels()
        with anki_vector.Robot(args.serial,
                               behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
            robot.camera.init_camera_feed()
            done = threading.Event()

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True).cuda()
            model.eval()

            tracker = RobotTracker()


            queue = Queue()

            robot.events.subscribe(on_new_raw_camera_image,
                                   events.Events.new_raw_camera_image, done,
                                   model, queue, tracker)
            print(
                "------ waiting for camera events, press ctrl+c to exit early ------")
            try:
                while(True):
                    detections = None
                    while not queue.empty():
                        detections = queue.get()

                    if detections:
                        image = np.array(detections['image'])

                        detections["mapped_labels"] = []
                        for k, label in enumerate(detections['labels']):
                            detections["mapped_labels"].append(
                                encoded_labels[label])
                            x0, y0, x1, y1  = detections['boxes'][k]
                            image = cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 0), 2)
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            cv2.putText(image,detections['mapped_labels'][k],(x0, y0), font, .5,(255,255,255),2,cv2.LINE_AA)

                        if detections['tracked_box'] is not None:
                            x, y, w, h = detections['tracked_box']
                            image = cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        cv2.imshow("IMAGE", image)
                        cv2.waitKey(10)
                        print(detections['mapped_labels'])

                    if not done.wait(timeout=5):
                        print("------ Did not receive a new camera image! ------")
            except KeyboardInterrupt:
                pass




if __name__ == "__main__":
    main()
