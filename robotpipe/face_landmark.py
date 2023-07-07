import robotpipe.insightface as insightface
from .face_detection import FaceDetection
from robotpipe.insightface.app.common import Face

class FaceLandmark:

    def __init__(self):
        self.detector = FaceDetection()

        self.model = insightface.model_zoo.get_model('best-landmark.onnx', download=False, download_zip=False)
        self.model.prepare(0)



    def predict(self,src):
        faces = self.detector.predict(src)

        for face in faces:
            self.model.get(src, face)

        return faces