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
import torchvision

def on_new_raw_camera_image(robot, event_type, event, done, model,
                            output_queue):
    image = event.image
    image_tensor = torchvision.transforms.ToTensor()(image).cuda()

    outputs = model([image_tensor])[0]

    scores = outputs["scores"]
    boxes = outputs["boxes"][scores > 0.5]
    labels = outputs["labels"][scores > 0.5]

    output_queue.put(dict(boxes=boxes.detach().cpu().numpy(),
                          labels=labels.detach().cpu().numpy(),
                          image=image))
    done.set()


def main():
        args = anki_vector.util.parse_command_args()
        with anki_vector.Robot(args.serial,
                               behavior_control_level=ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY) as robot:
            robot.camera.init_camera_feed()
            done = threading.Event()

            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
                pretrained=True).cuda()
            model.eval()

            queue = Queue()

            robot.events.subscribe(on_new_raw_camera_image,
                                   events.Events.new_raw_camera_image, done,
                                   model, queue)
            print(
                "------ waiting for camera events, press ctrl+c to exit early ------")
            try:
                while(True):
                    detections = None
                    while not queue.empty():
                        detections = queue.get()

                    if detections:
                        print(detections['labels'])

                    if not done.wait(timeout=5):
                        print("------ Did not receive a new camera image! ------")
            except KeyboardInterrupt:
                pass




if __name__ == "__main__":
    main()
