# Continuous motoring system in real time for streaming cameras 

This repo provides several samples about how to develop a continuous monitoring system for streaming cameras. Mainly, the frames from the streaming cameras are processed (ex. person tracking, object detection, face detection) and the results are sent to [Devo](www.devo.com) platform for monitoring. 
 
The main components are:

*Openvino/Opencv Libs* -> to process the frames generating an event with the results (bounding box, performance, errors, etc.) 

*Devo* -> for events monitoring

*Docker* -> for debugging and application deploy

The docker file build a basic image common to all samples. This basic image set the OpenVino environment and the Devo sdk library. 

Openvino toolkit version: R3.1

## How It Works

docker build -t openvinodevosdk/basic .

## Samples:

### Requierements

Like this demo generate events to be sent to Devo, it's required to download from the platform the authentication certs. See ___ for more info. 
The user certificates must be saved to the CERTS folder 

![image info](./readme_imgs/certs.png)

### Person tracking
Person tracking through multiple cameras. The person tracker demo is used to build an event log with the eventdate, personId, cameraId and the bounding box features at the current frame. This info allow the use of statistics for reporting (person counter, marketing campaign analysis) or to allow the use of alerts (person intrusion, anomaly behavior).

#### Example of usage

##### Build image 
cd /person_tracker
./build.sh -> build openvinodevosdk/person_tracker docker image 

##### run help
docker run --rm -it openvinodevosdk/person_tracker -h

### Object Detector
After being processed a frame, an event is sent for each detected object. The event log is related to the detection eventdate, object class, bounding box.
 
#### Example of usage

##### Build image 
cd /object_detector
./build.sh -> build openvinodevosdk/object_detection docker image 

##### run help
docker run --rm -it openvinodevosdk/object_detection -h

