import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle


class FaceMeshDetector:
    def __init__(self):
        model_asset_path = os.path.join(current_dir, "../checkpoint/face_landmarker_v2_with_blendshapes.task")
        base_options = python.BaseOptions(model_asset_path = model_asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        model_asset_path = os.path.join(current_dir, "../checkpoint/bs_dict.pkl")

        with open(model_asset_path, "rb") as f:
            self.bs_model = pickle.load(f)

    def detect(self, cv_mat):
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(cv_mat))
        mp_result = self.detector.detect(rgb_frame)
        final_morph = np.zeros([478, 3])
        if len(mp_result.face_landmarks) >= 1 and len(mp_result.face_blendshapes) >= 1:
            self.landmarks = mp_result.face_landmarks[0]
            self.blendshapes = mp_result.face_blendshapes[0]
            self.facial_matrixes = mp_result.facial_transformation_matrixes[0]
            # ss = {}
            for face_blendshapes_category in self.blendshapes:
                if "neutral" not in face_blendshapes_category.category_name and face_blendshapes_category.category_name in self.bs_model.keys() :
                    final_morph += face_blendshapes_category.score * self.bs_model[face_blendshapes_category.category_name]
                # if "jaw" in face_blendshapes_category.category_name or "mouthClose" in face_blendshapes_category.category_name:
                #     ss[face_blendshapes_category.category_name] = face_blendshapes_category.score
            # print(ss)
        else:
            [self.landmarks, self.blendshapes, self.facial_matrixes] = [None, None, np.eye(4)]
        return self.landmarks, self.blendshapes, final_morph, self.facial_matrixes

# https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/face_landmarker/python
def draw_landmarks_on_image(rgb_image, face_landmarks):
  face_landmarks_list = [face_landmarks]
  annotated_image = np.copy(rgb_image)
  from mediapipe.framework.formats import landmark_pb2
  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))
  print(face_blendshapes_names, len(face_blendshapes_names))
  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

if __name__ == "__main__":
    import cv2
    face_detector = FaceMeshDetector()
    cv_mat = cv2.imread(r"F:/1692399104478.png")[:, :, ::-1]
    import time

    start_time = time.time()
    for i in range(1, 2):
        landmarks, blendshapes, final_morph, facial_matrixes = face_detector.detect(cv_mat)
        print(i, (time.time() - start_time) / i)
        print(final_morph.shape)
        # STEP 5: Process the detection result. In this case, visualize it.
        annotated_image = draw_landmarks_on_image(cv_mat, landmarks)
        print(i, (time.time() - start_time) / i)
        cv2.imshow("sss", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(-1)
    plot_face_blendshapes_bar_graph(blendshapes)

    print(facial_matrixes)
    print(np.linalg.det(facial_matrixes[:3, :3]))