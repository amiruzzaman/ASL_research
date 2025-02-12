import { decodeAsync } from "@msgpack/msgpack";

import * as THREE from "three";

export type Vec3 = [number, number, number];
export type Hand = Vec3[];
export type Frame = [Hand, Hand, Hand];
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

const seperatePhrase = (input: string): string[] => {
  // TODO: Simple splitting for now
  return input.toLowerCase().split(" ");
};

export type TranslationRequest = {
  words: string[];
  dataMap: Record<string, WordData>;
};

const createRequest = async (phrase: string): Promise<TranslationRequest> => {
  const words = seperatePhrase(phrase);

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
  camera.position.z = 0.75;
  camera.position.x = -0.5;
  camera.position.y = -0.5;
  return { render: renderer, cam: camera, scene };
};

const fixPos = ([x, y, z]: Vec3): [number, number, number] => {
  return [-x, -y, -z];
};

const basicMat = (color: number) => new THREE.MeshBasicMaterial({ color });

const createPoint = (scene: THREE.Scene, pos: Vec3, mat: THREE.MeshBasicMaterial) => {
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.006), mat);
  scene.add(sphere);
  sphere.position.set(...fixPos(pos));
  return sphere;
};

const lineConnections: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 4],
  //[3, 5],
  [0, 5],
  [5, 6],
  [6, 7],
  [7, 8],
  [5, 9],
  [9, 10],
  [10, 11],
  [11, 12],
  [9, 13],
  [13, 14],
  [14, 15],
  [15, 16],
  [13, 17],
  [17, 18],
  [18, 19],
  [19, 20],
  [17, 0],
];

const updatePoint = (point: THREE.Mesh, pos: Vec3) => {
  point.position.set(...fixPos(pos));
};

const lineMat = (color: number) => new THREE.LineBasicMaterial({ color });

const connectPoints = (
  scene: THREE.Scene,
  pointA: Vec3,
  pointB: Vec3,
  mat: THREE.LineBasicMaterial,
) => {
  const geom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(...pointA),
    new THREE.Vector3(...pointB),
  ]);
  const line = new THREE.Line(geom, mat);
  scene.add(line);
  return line;
};

const updateLine = (line: THREE.Line, newA: Vec3, newB: Vec3) => {
  line.geometry.setFromPoints([new THREE.Vector3(...newA), new THREE.Vector3(...newB)]);
};

type RenderHand = [THREE.Mesh[], THREE.Line[]];

const fallBackHand = [
  ...Array(21)
    .keys()
    .map((_) => [5, 5, 5] as Vec3),
];
const fallBackFace = [
  ...Array(468)
    .keys()
    .map((_) => [5, 5, 5] as Vec3),
];

const createHand = (
  scene: THREE.Scene,
  hand: Hand,
  color: number,
  connectArray: [number, number][],
): RenderHand => [
  hand.map((p) => createPoint(scene, p, basicMat(color))),
  connectArray.map(([pointA, pointB]) =>
    connectPoints(scene, fixPos(hand[pointA]), fixPos(hand[pointB]), lineMat(color)),
  ),
];

const updateHand = (currentHand: RenderHand, newHand: Hand, connectArray: [number, number][]) => {
  (newHand ?? []).forEach((pos, i) => {
    updatePoint(currentHand[0][i], pos);
  });
  currentHand[1].forEach((line, i) => {
    updateLine(line, fixPos(newHand[connectArray[i][0]]), fixPos(newHand[connectArray[i][1]]));
  });
};

const deleteHand = (scene: THREE.Scene, hand: RenderHand) => {
  hand[0].forEach((p) => scene.remove(p));
  hand[1].forEach((p) => scene.remove(p));
};

const createScene = (
  scene: THREE.Scene,
  [lh, rh, f]: Frame | [null, null, null],
): [RenderHand, RenderHand, RenderHand] => [
  createHand(scene, lh ?? fallBackHand, 0x00ff00, lineConnections),
  createHand(scene, rh ?? fallBackHand, 0x0000ff, lineConnections),
  createHand(scene, f ?? fallBackFace, 0xff0000, []),
];

const updateScene = (
  [currentLh, currentRh, currentF]: [RenderHand, RenderHand, RenderHand],
  [newLh, newRh, newF]: Frame,
) => {
  updateHand(currentLh, newLh ?? fallBackHand, lineConnections);
  updateHand(currentRh, newRh ?? fallBackHand, lineConnections);
  updateHand(currentF, newF ?? fallBackFace, []);
};

const deleteScene = (scene: THREE.Scene, [lh, rh, f]: [RenderHand, RenderHand, RenderHand]) => {
  deleteHand(scene, lh);
  deleteHand(scene, rh);
  deleteHand(scene, f);
};

type AnimationContext = {
  currentWord: number;
  currentFrame: number;
  paused: boolean;
  currentScene: [RenderHand, RenderHand, RenderHand];
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
    currentScene: createScene(ctx.scene, [null, null, null]),
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
          updateScene(animContext.currentScene, currFrameData);
        }
      } else {
        updateScene(animContext.currentScene, currFrameData);
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
