import express from "express";
import fs from "fs";
import path from "path";
import * as log from "@vladmandic/pilogger";
import * as tf from "@tensorflow/tfjs-node";
import * as faceapi from "@vladmandic/face-api";
import multer from "multer";

const app = express();
const storage = multer.memoryStorage();
const upload = multer({
  storage: storage,
});

const distanceThreshold = 0.6;
const modelPath = "models";

async function initFaceAPI() {
  await faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath);
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath);
}

async function registerImage(inputFile: string, label: string) {
  if (
    !inputFile.toLowerCase().endsWith("jpg") &&
    !inputFile.toLowerCase().endsWith("png") &&
    !inputFile.toLowerCase().endsWith("gif")
  )
    return;
  const descriptor = await getDescriptors(inputFile);

  if (!descriptor) return;
  const labeledFaceDescriptor = new faceapi.LabeledFaceDescriptors(label, [
    descriptor,
  ]);
  return labeledFaceDescriptor;
}

async function getDescriptors(imageFile: string | Uint8Array) {
  let buffer: Buffer | Uint8Array;
  if (typeof imageFile == "string") {
    buffer = fs.readFileSync(imageFile);
  } else {
    buffer = imageFile;
  }
  const tensor: any = tf.node.decodeImage(buffer, 3);
  const faces = await faceapi
    .detectSingleFace(tensor)
    .withFaceLandmarks()
    .withFaceDescriptor();
  tf.dispose(tensor);
  return faces?.descriptor;
}

async function findBestMatch(
  inputFile: string | Float32Array | Uint8Array,
  labeledFaceDescriptors: faceapi.LabeledFaceDescriptors[]
) {
  const matcher = new faceapi.FaceMatcher(
    labeledFaceDescriptors,
    distanceThreshold
  );
  const descriptor =
    typeof inputFile === "string" || inputFile instanceof Uint8Array
      ? await getDescriptors(inputFile)
      : inputFile;

  if (!descriptor) return;
  const match = await matcher.findBestMatch(descriptor);
  return match;
}

async function registerFace(faceUser: string) {
  const userPath = path.join(__dirname, `labeled_images/${faceUser}`);
  if (!fs.existsSync(userPath)) {
    throw {
      status: 404,
      error: "User not found",
    };
  }

  const dir = fs.readdirSync(userPath);

  const labeledFaceDescriptors: faceapi.LabeledFaceDescriptors[] = [];

  for (const file of dir) {
    const labeledFaceDescriptor = await registerImage(
      path.join(__dirname, `labeled_images/${faceUser}/${file}`),
      faceUser
    );
    if (labeledFaceDescriptor) {
      labeledFaceDescriptors.push(labeledFaceDescriptor);
    }
  }
  return labeledFaceDescriptors;
}

app.post("/upload", upload.single("face"), async function (req, res) {
  const file_buffer = req?.file?.["buffer"];

  if (!file_buffer) return res.sendStatus(404);

  const { user } = req.body;

  try {
    const labeledFaceDescriptors = await registerFace(user);

    if (!labeledFaceDescriptors.length)
      return res.send("User pictures are not available");

    const uintArray = new Uint8Array(file_buffer);

    const bestMatch = await findBestMatch(uintArray, labeledFaceDescriptors);
    if (!bestMatch || bestMatch?.label == "unknown") {
      return res.send({
        user,
        message: "Face is not matched",
      });
    } else {
      return res.send({
        user,
        message: `Face matched with ${Math.round(
          (1 - bestMatch.distance) * 100
        )}%`,
      });
    }
  } catch (error: any) {
    return res.status(error.status || 500).json(error.error || error);
  }
});

app.listen(4724, () => {
  console.log(`Server started at http://localhost:4724`);
  initFaceAPI();
});
