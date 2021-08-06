import cv2
import os
import dlib
import math

'''Method to update the drone memory with new detection data for centroid tracking'''
def update_drone_memory(conscious_memory, bbox, ID):
    centroid = (int(bbox.top() + ((bbox.bottom() - bbox.top()) / 2)), int(bbox.left() + ((bbox.right() - bbox.left()) / 2)));
    next_centroid_probable_region_bottom = bbox.bottom();
    next_centroid_probable_region_right = bbox.right();
    next_centroid_probable_region_top = bbox.top();
    next_centroid_probable_region_left = bbox.left();
    next_centroid_probable_region = (next_centroid_probable_region_left, next_centroid_probable_region_top, next_centroid_probable_region_right, next_centroid_probable_region_bottom);    
    conscious_memory[ID] = [bbox, next_centroid_probable_region, centroid];
    return conscious_memory;

'''Method to scan the drone memory for existing detection and return ID new detection
is found'''
def scan_drone_memory(conscious_memory, subconscious_memory, bbox):
    centroid = (int(bbox.top() + ((bbox.bottom() - bbox.top()) / 2)), int(bbox.left() + ((bbox.right() - bbox.left()) / 2)));
    if(len(conscious_memory) != 0):
        for i in list(conscious_memory.values()):
            if((centroid[0] >= i[1][1] and centroid[0] <= i[1][3]) and (centroid[1] >= i[1][0] and centroid[1] <= i[1][2])):
                return list(conscious_memory.keys())[list(conscious_memory.values()).index(i)];
    if(len(subconscious_memory) != 0):
        for i in list(subconscious_memory.values()):
            if((centroid[0] >= i[1][1] and centroid[0] <= i[1][3]) and (centroid[1] >= i[1][0] and centroid[1] <= i[1][2])):
                return list(subconscious_memory.keys())[list(subconscious_memory.values()).index(i)];
    return -1;

'''Method to refresh the drone memory to avoid redundant detections'''
def refresh_drone_memory(conscious_memory, subconscious_memory):
    i = 0; j = 0;
    while True:
        if(i < len(conscious_memory.values()) - 1):
            j = i + 1;
            while True:
                if(j < len(conscious_memory.values())):
                    if((list(conscious_memory.values())[i][2][0] >= list(conscious_memory.values())[j][1][1] and list(conscious_memory.values())[i][2][0] <= list(conscious_memory.values())[j][1][3]) and (list(conscious_memory.values())[i][2][1] >= list(conscious_memory.values())[j][1][0] and list(conscious_memory.values())[i][2][1] <= list(conscious_memory.values())[j][1][2])):
                        subconscious_memory.update({list(conscious_memory.keys())[i] : conscious_memory.pop(list(conscious_memory.keys())[i])});
                    j += 1;
                else:
                    break;
            i += 1;
        else:
            break;
    return conscious_memory;

'''Initialize the trained HOG and SVM detector'''              
detector = dlib.simple_object_detector('Car_Detector_1 (C = 7.605).svm');

'''Initialize the Input Video''' 
cap = cv2.VideoCapture('DRONE-SURVEILLANCE-CONTEST-VIDEO-(Resized).mp4');

'''Initialize conscious memory for drone'''
drone_cons_mem = {};

'''Initialize subconscious memory for drone'''
drone_subcon_mem = {};

'''Set upper memory boundary to store the older detections in subconscious
memory of drone'''
upper_mem_bound = 310;

'''Set lower memory boundary for newer detections'''
lower_mem_bound = 640;
c = 0;

while True:
    ret, f = cap.read();
    if(ret):
        detections = detector(f);
        for detection in detections:
            x1 = int(detection.left());
            y1 = int(detection.top());
            x2 = int(detection.right());
            y2 = int(detection.bottom());
            '''Check if the detections are within the valid memory boundaries.
               Update the newer detections in drone conscious memory and
               transfer the older detections in drone subconscious memory.'''
            if(y2 > upper_mem_bound and y1 < lower_mem_bound):
                if(len(drone_cons_mem) == 0):
                    c += 1;
                    drone_cons_mem = update_drone_memory(drone_cons_mem, detection, c);
                    cv2.putText(f, str(c), (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2);
                else:
                    ID = scan_drone_memory(drone_cons_mem, drone_subcon_mem, detection);
                    if(ID in drone_cons_mem.keys()):
                        drone_cons_mem = update_drone_memory(drone_cons_mem, detection, ID);
                        cv2.putText(f, str(ID), (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2);
                    else:
                        if(ID not in drone_subcon_mem.keys()):
                            c += 1;
                            drone_cons_mem = update_drone_memory(drone_cons_mem, detection, c);
                            cv2.putText(f, str(c), (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2);
            else:
                if(y2 <= upper_mem_bound):
                    if(len(drone_cons_mem) != 0):
                        ID = scan_drone_memory(drone_cons_mem, drone_subcon_mem, detection);
                        if(ID in drone_cons_mem.keys()):
                            drone_subcon_mem.update({ID : drone_cons_mem.pop(ID)});
            cv2.rectangle(f, (x1, y1), (x2, y2), (0, 255, 0), 2);
        if(c > 99):
            cv2.putText(f, str(c), (1130, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5);
        else:
            cv2.putText(f, str(c), (1150, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5);
        cv2.putText(f, "PAVAN", (600, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5);    
        cv2.imshow('f', f);
        '''Refresh the drone conscious memory to transfer redundant detections
           in drone subconscious memory'''
        refresh_drone_memory(drone_cons_mem, drone_subcon_mem);
        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break;
    else:
        break;

cap.release();
#cv2.destroyAllWindows();
