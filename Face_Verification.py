# Copyright © 2020 BYTEPAL AI, LLC And Its Affiliates. All rights reserved.
import face_recognition
import cv2
import time


def draw_face(path,start_point, end_point):
    image = cv2.imread(path)
    window_name = 'Image'
    color = (255, 0, 0)
    thickness = 2
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)


def recognition_model(photo1, photo2):
    picture_of_me = face_recognition.load_image_file(photo1)
    face_locations1 = face_recognition.face_locations(picture_of_me, model="hog") # test vs CNN on the Digital Ocean instance
    my_face_encoding = face_recognition.face_encodings(picture_of_me, face_locations1)
    # my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!
    unknown_picture = face_recognition.load_image_file(photo2)
    face_locations2 = face_recognition.face_locations(unknown_picture, model="hog") # Test vs CNN on the Digital Ocean instance
    unknown_face_encoding = face_recognition.face_encodings(unknown_picture, face_locations2)[0]
    # Now we can see the two face encodings are of the same person with `compare_faces`!
    #results = face_recognition.compare_faces(my_face_encoding, unknown_face_encoding) # Compare with all the faces in the picture
    results = face_recognition.compare_faces(my_face_encoding, unknown_face_encoding)
    print(results)
    index = 0

    for result in results:
        if result == True:
            face_location = face_locations1[index]
            top, right, bottom, left = face_location
            print("coordinates of matched face", index,  face_location) # Answer , 0, then coordinates of Paul Face
            # css (top, right, bottom, left)
            top_left = (left, top)
            bottom_right = (right, bottom)
            #draw_face(photo1, top_left, bottom_right)
            output = {"found":True, "coordinates":{"top":top,"right":right,"bottom":bottom,"left":left}}
            return output
        index += 1

    output = {"found":False, "coordinates":{"top":None,"right":None,"bottom":None,"left":None}}
    return output
