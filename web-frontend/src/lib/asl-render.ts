import "./dispose-poly.ts";

import { decodeAsync } from "@msgpack/msgpack";

import handConnections from "./hand_connections.json" with { type: "json" };
import faceConnections from "./face_connections.json" with { type: "json" };
import poseConnections from "./pose_connections.json" with { type: "json" };

import * as THREE from "three";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

export type Vec3 = [number, number, number];
export type Hand = Vec3[];
export type Frame = [Hand | null, Hand | null, Hand | null, Hand | null];
export type WordData = Frame[];

const backendEndpoint = import.meta.env.PUBLIC_BACKEND_HOST ?? "";
const wordEndpoint = `${backendEndpoint}/api/word`;

const fetchWord = async (word: string): Promise<WordData | null> => {
  const response = await fetch(`${wordEndpoint}/${word}`);
  if (response.ok && response.body !== null) {
    return decodeAsync(response.body) as Promise<WordData>;
  } else if (response.status === 404) {
    return null;
  } else {
    console.error(`Backend returned ${response.status}: ${await response.text()}`);
    return null;
  }
};

const separatePhrase = (input: string): string[] => {
  // TODO: Simple splitting for now
  return input.toLowerCase().split(" ");
};

export type TranslationRequest = {
  words: string[];
  dataMap: Record<string, WordData>;
};

const createRequest = async (phrase: string): Promise<TranslationRequest> => {
  const words = separatePhrase(phrase);

  const dataMap: Record<string, WordData> = {};
  const failedWords: string[] = [];

  for (const word of words) {
    if (!failedWords.includes(word) && !(word in dataMap)) {
      const data = await fetchWord(word);

      if (data) {
        dataMap[word] = data;
      } else {
        failedWords.push(word);
      }
    }
  }

  return {
    words,
    dataMap,
  };
};

export type ThreeContext = {
  render: THREE.WebGLRenderer;
  scene: THREE.Scene;
  cam: THREE.Camera;
  anim?: AnimationContext;
};

const initThree = (parent: HTMLElement): ThreeContext => {
  const width = parent.offsetWidth;
  const height = parent.offsetHeight;
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 2000);

  const onParentResize = (entries: ResizeObserverEntry[]) => {
    const lastItem = entries[entries.length - 1];
    const { inlineSize: width, blockSize: height } = lastItem.contentBoxSize[0];
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
  };

  const observer = new ResizeObserver(onParentResize);
  observer.observe(parent);

  const renderer = new THREE.WebGLRenderer({ alpha: true });
  renderer.setSize(width, height);
  parent.appendChild(renderer.domElement);
  camera.position.z = 1;
  camera.position.x = 0;
  camera.position.y = 0.5;
  return { render: renderer, cam: camera, scene };
};

const fixPos = ([x, y, z]: Vec3): Vec3 => {
  return [-x, -y, -z];
};

const componentWise = (vecA: Vec3, vecB: Vec3, f: (c1: number, c2: number) => number): Vec3 => [
  f(vecA[0], vecB[0]),
  f(vecA[1], vecB[1]),
  f(vecA[2], vecB[2]),
];

const addVec = (vecA: Vec3, vecB: Vec3): Vec3 => componentWise(vecA, vecB, (a, b) => a + b);
const subtractVec = (vecA: Vec3, vecB: Vec3): Vec3 => componentWise(vecA, vecB, (a, b) => a - b);

const transformToPose = (
  posePartRoot: number,
  pose: Vec3[],
  partRoot: number,
  part: Vec3[] | null,
): Vec3[] => {
  const rootInPose = pose?.[posePartRoot] ?? ([0, 0, 0] as Vec3);
  const rootInPart = part?.[partRoot] ?? ([0, 0, 0] as Vec3);
  return part?.map?.((p) => fixPos(addVec(subtractVec(p, rootInPart), rootInPose))) ?? [];
};

// Points we don't want in pose since they're often wrong or unneeded in our context.
const filterPosePoints = [26, 25, 27, 28, 32, 30, 29, 31, 21, 19, 17, 22, 20, 18];

// Positions all points relative to the pose landmarks
const transformFrame = ([lh, rh, face, pose]: Frame): Frame => [
  pose && lh ? transformToPose(16, pose, 0, lh) : null,
  pose && rh ? transformToPose(15, pose, 0, rh) : null,
  pose && face ? transformToPose(0, pose, 0, face) : null,
  pose?.map?.(fixPos)?.map((p, i) => (filterPosePoints.includes(i) ? fallBackPose[i] : p)) ?? null,
];

const basicMat = (color: number) => new THREE.MeshBasicMaterial({ color });

const createPoint = (scene: THREE.Scene, pos: Vec3, mat: THREE.MeshBasicMaterial) => {
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.003), mat);
  scene.add(sphere);
  sphere.position.set(...pos);
  return sphere;
};

const updatePoint = (point: THREE.Mesh, pos: Vec3) => {
  point.position.set(...pos);
};

const lineMat = (color: number) => new LineMaterial({ color });

const connectPoints = (scene: THREE.Scene, pointA: Vec3, pointB: Vec3, mat: LineMaterial) => {
  const geom = new LineSegmentsGeometry().setPositions([...pointA, ...pointB]);
  const line = new LineSegments2(geom, mat);
  scene.add(line);
  return line;
};

const updateLine = (line: LineSegments2, newA: Vec3, newB: Vec3) => {
  line.geometry.setPositions([...newA, ...newB]);
};

type RenderHand = [THREE.Mesh[], LineSegments2[]];

const fallBackHand = [
  ...Array(21)
    .keys()
    .map(() => [5, 5, 5] as Vec3),
];
const fallBackFace = [
  ...Array(478)
    .keys()
    .map(() => [5, 5, 5] as Vec3),
];
const fallBackPose = [
  ...Array(33)
    .keys()
    .map(() => [5, 5, 5] as Vec3),
];

const createHand = (
  scene: THREE.Scene,
  hand: Hand,
  color: number,
  connectArray: Conn,
): RenderHand => [
  hand.map((p) => createPoint(scene, p, basicMat(color))),
  connectArray.map(([pointA, pointB]) =>
    connectPoints(scene, hand[pointA], hand[pointB], lineMat(color)),
  ),
];

const updateHand = (currentHand: RenderHand, newHand: Hand, connectArray: [number, number][]) => {
  (newHand ?? []).forEach((pos, i) => {
    updatePoint(currentHand[0][i], pos);
  });
  currentHand[1].forEach((line, i) => {
    const [pointA, pointB] = connectArray[i].map((j) => newHand[j]);
    if (pointA && pointB) {
      updateLine(line, pointA, pointB);
    }
  });
};

const deleteHand = (scene: THREE.Scene, hand: RenderHand) => {
  hand[0].forEach((p) => scene.remove(p));
  hand[1].forEach((p) => scene.remove(p));
};

type Conn = [number, number][];
type SceneState = [RenderHand, RenderHand, RenderHand, RenderHand];

const createScene = (scene: THREE.Scene): SceneState => [
  createHand(scene, fallBackHand, 0x00ff00, handConnections as Conn), // Left Hand
  createHand(scene, fallBackHand, 0x4aefdc, handConnections as Conn), // Right Hand
  createHand(scene, fallBackFace, 0xff0000, faceConnections as Conn), // Face
  createHand(scene, fallBackPose, 0xffff00, poseConnections as Conn), // Pose
];

const updateScene = (
  [currentLh, currentRh, currentF, currentPose]: SceneState,
  [newLh, newRh, newF, newPose]: Frame,
) => {
  updateHand(currentLh, newLh ?? fallBackHand, handConnections as Conn);
  updateHand(currentRh, newRh ?? fallBackHand, handConnections as Conn);
  updateHand(currentF, newF ?? fallBackFace, faceConnections as Conn);
  updateHand(currentPose, newPose ?? fallBackPose, poseConnections as Conn);
};

const deleteScene = (scene: THREE.Scene, [lh, rh, f, p]: SceneState) => {
  deleteHand(scene, lh);
  deleteHand(scene, rh);
  deleteHand(scene, f);
  deleteHand(scene, p);
};

type AnimationContext = {
  currentWord: number;
  currentFrame: number;
  paused: boolean;
  currentScene: SceneState;
};

const prepareAnim = (
  ctx: ThreeContext,
  req: TranslationRequest,
  onWordChange?: (index: number) => void,
) => {
  const animContext: AnimationContext = {
    currentWord: 0,
    currentFrame: 0,
    paused: false,
    currentScene: createScene(ctx.scene),
  };

  ctx.anim = animContext;

  const wordLen = req.words.length;
  const waitTime = 300;

  // == MAIN ANIMATION LOOP ==
  const mainLoop = () => {
    if (animContext.paused) {
      ctx.render.render(ctx.scene, ctx.cam);
      return;
    }

    const currWordPos = animContext.currentWord % wordLen;
    const currWord = req.words[currWordPos];
    const currWordData = req.dataMap[currWord] ?? [];

    const currFrame = animContext.currentFrame;

    if (currFrame >= currWordData.length) {
      // Next Word
      animContext.paused = true;
      setTimeout(() => {
        animContext.currentWord += 1;
        animContext.currentFrame = 0;
        animContext.paused = false;
      }, waitTime);
    } else {
      // Still on this word
      const currFrameData = currWordData[animContext.currentFrame];

      if (currFrame === 0) {
        onWordChange?.(currWordPos);
        if (currFrameData === undefined) {
          // Our word is unknown, skip it
          animContext.currentWord += 1;
          return;
        } else {
          updateScene(animContext.currentScene, transformFrame(currFrameData));
        }
      } else {
        updateScene(animContext.currentScene, transformFrame(currFrameData));
      }
      animContext.currentFrame += 1;
    }

    ctx.render.render(ctx.scene, ctx.cam);
  };

  return mainLoop;
};

export const prepareCanvas = (canvasParent: HTMLElement): ThreeContext => initThree(canvasParent);

export const parseAndCreateRequest = (input: string): Promise<TranslationRequest> =>
  createRequest(input);

export const renderAsl = (
  ctx: ThreeContext,
  req: TranslationRequest,
  onWordChange?: (index: number) => void,
) => {
  if (ctx.anim !== undefined) {
    ctx.anim.paused = true;
    deleteScene(ctx.scene, ctx.anim.currentScene);
    ctx.render.setAnimationLoop(null);
  }
  const anim = prepareAnim(ctx, req, onWordChange);
  ctx.render.setAnimationLoop(anim);
};
