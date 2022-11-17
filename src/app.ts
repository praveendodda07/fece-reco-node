import express from "express";
import fs from "fs";
import path from "path";
import * as log from "@vladmandic/pilogger";
import * as tf from "@tensorflow/tfjs-node";
import * as faceapi from "@vladmandic/face-api";

const app = express();

const distanceThreshold = 0.6;
const modelPath = "models";
const labeledFaceDescriptors: any[] = [];

// Test image came from payload
const test_image = "test-praveen-1.png";

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
  log.data("Registered:", inputFile);
  const descriptor = await getDescriptors(inputFile);

  if (!descriptor) return;
  const labeledFaceDescriptor = new faceapi.LabeledFaceDescriptors(label, [
    descriptor,
  ]);
  labeledFaceDescriptors.push(labeledFaceDescriptor);
}

async function getDescriptors(imageFile: string) {
  const buffer = fs.readFileSync(imageFile);
  const tensor: any = tf.node.decodeImage(buffer, 3);
  const faces = await faceapi
    .detectSingleFace(tensor)
    .withFaceLandmarks()
    .withFaceDescriptor();
  tf.dispose(tensor);
  return faces?.descriptor;
}

async function findBestMatch(inputFile: string) {
  const matcher = new faceapi.FaceMatcher(
    labeledFaceDescriptors,
    distanceThreshold
  );
  const descriptor = await getDescriptors(inputFile);
  if (!descriptor) return;
  const match = await matcher.findBestMatch(descriptor);
  return match;
}

async function main() {
  log.header();

  await initFaceAPI();

  const labels = ["Praveen", "Steve Jobs"];
  for (const faceUser of labels) {
    const dir = fs.readdirSync(
      path.join(__dirname, `labeled_images/${faceUser}`)
    );
    for (const f of dir)
      await registerImage(
        path.join(__dirname, `labeled_images/${faceUser}/${f}`),
        faceUser
      );
  }

  log.info(
    "Comparing:",
    test_image,
    "Descriptors:",
    labeledFaceDescriptors.length
  );
  if (labeledFaceDescriptors.length > 0) {
    const bestMatch = await findBestMatch(
      path.join(__dirname, `test_images/${test_image}`)
    ); // find best match to all registered images
    log.data("Match:", bestMatch);
  } else {
    log.warn("No registered faces");
  }
}

app.listen(4724, () => {
  console.log(`Server started at http://localhost:4724`);
  main();
});
