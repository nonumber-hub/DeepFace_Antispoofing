from deepface import DeepFace

DeepFace.stream("tests/dataset", enable_face_analysis=True, anti_spoofing=True)  # opencv
# DeepFace.stream("dataset", detector_backend = 'opencv')
# DeepFace.stream("dataset", detector_backend = 'ssd')
# DeepFace.stream("dataset", detector_backend = 'mtcnn')
# DeepFace.stream("dataset", detector_backend = 'dlib')
# DeepFace.stream("dataset", detector_backend = 'retinaface')
