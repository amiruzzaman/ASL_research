import "./dispose-poly.ts";

import { decodeAsync, decodeMultiStream } from "@msgpack/msgpack";

import handConnections from "./hand_connections.json" with { type: "json" };
import faceConnections from "./face_connections.json" with { type: "json" };
import poseConnections from "./pose_connections.json" with { type: "json" };
import {
  type Data,
  drawConnectors,
  type DrawingOptions,
  drawLandmarks,
  lerp,
} from "@mediapipe/drawing_utils";

export type Vec3 = { x: number; y: number; z?: number };
export type Part = Vec3[];
export type Frame = { face: Part; pose: Part; hands: [Part, Part] };
export type WordData = { fps: number | null; frames: Frame[] };

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

type FpsMarker = { fps: number | null };
type StreamedFrame = FpsMarker | Frame;

export const receiveWordStreamed = async (resp: Response): Promise<WordData | null> => {
  if (resp.ok && resp.body !== null) {
    const frames = [];
    let fps = null;
    for await (const frame of decodeMultiStream(resp.body) as AsyncGenerator<StreamedFrame>) {
      if (fps === null && Object.hasOwn(frame, "fps")) {
        fps = (frame as FpsMarker).fps;
      } else {
        frames.push(frame as Frame);
      }
    }
    return { fps, frames };
  } else {
    // TODO: Error Handle
    console.error(resp);
    return null;
  }
};

export type TranslationRequest = {
  words: string[];
  dataMap: Record<string, WordData>;
};

export type WordProgress = {
  currentWord: number;
  totalWords: number;
};

const getGlossTerms = async (phrase: string): Promise<string[]> => {
  const resp = await fetch(`${backendEndpoint}/api/gloss`, {
    body: phrase,
    method: "POST",
  });
  if (resp.ok && resp.body !== null) {
    return decodeAsync(resp.body) as Promise<string[]>;
  } else {
    console.error("Failed to get gloss terms", resp.status, await resp.text());
    return [];
  }
};

export const createRequest = async (
  phrase: string,
  progressCallback?: (payload: WordProgress) => void,
): Promise<TranslationRequest> => {
  const words = await getGlossTerms(phrase);

  const dataMap: Record<string, WordData> = {};
  const failedWords: string[] = [];

  const totalWords = words.length;
  let currentWord = 1;

  for (const word of words) {
    progressCallback?.({ currentWord, totalWords });
    currentWord++;
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

type AnimationContext = {
  req: TranslationRequest;
  currentWord: number;
  currentFrame: number;
  paused: boolean;
  currentFps: number;
  lastRender: number;
  wordCallback?: (idx: number) => void;
};

export type RenderContext = {
  canvas: HTMLCanvasElement;
  dimensions: [number, number];
  anim?: AnimationContext;
};

type Connections = [number, number][];

const renderPart = (
  canvas: CanvasRenderingContext2D,
  part: Part,
  connections: Connections,
  opts: DrawingOptions,
) => {
  drawConnectors(canvas, part, connections, opts);
  drawLandmarks(canvas, part, opts);
};

const colorSchemeQuery = window.matchMedia("(prefers-color-scheme: dark)");

type PartColors = { pose: string; face: string; hands: [string, string] };
type Palette = { light: PartColors; dark: PartColors };

const PART_COLORS = {
  light: { pose: "#38323b", face: "#644185", hands: ["#851409", "#094585"] },
  dark: { pose: "#b6a6bd", face: "#8b46e0", hands: ["#de3131", "#3182de"] },
} as Palette;

let partColors = colorSchemeQuery.matches ? PART_COLORS["dark"] : PART_COLORS["light"];

colorSchemeQuery.addEventListener("change", (e) => {
  partColors = e.matches ? PART_COLORS["dark"] : PART_COLORS["light"];
});

const renderFrame = (ctx: RenderContext, canvas: CanvasRenderingContext2D, frame: Frame) => {
  if (!canvas) {
    console.error("Failed to get canvas context!");
    return;
  }

  // Clearing Last Frame
  //canvas.globalCompositeOperation = "destination-over";
  //canvas.fillStyle = "rgb(0 0 0 / 40%)";
  canvas.save();
  canvas.clearRect(0, 0, ...ctx.dimensions);

  const processPoint = (data: Data) => lerp(data.from!.z!, -0.15, 0.1, 1, 0.1);
  const poseExclude = [
    26, 25, 27, 28, 32, 30, 29, 31, 21, 19, 17, 22, 20, 18, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 9,
  ];

  const globalOpts = {
    lineWidth: 1,
    radius: processPoint,
  };

  // Drawing Each Part in the Frame
  // Pose -> Face -> Hands as hands should go over face and pose (usually hands are in front of them)
  renderPart(canvas, frame.pose, poseConnections as Connections, {
    color: partColors["pose"],
    ...globalOpts,
    radius: (data) => (poseExclude.includes(data.index!) ? 0 : processPoint(data)),
  });
  renderPart(canvas, frame.face, faceConnections as Connections, {
    color: partColors["face"],
    ...globalOpts,
  });
  renderPart(canvas, frame.hands[0], handConnections as Connections, {
    color: partColors["hands"][0],
    ...globalOpts,
  });
  renderPart(canvas, frame.hands[1], handConnections as Connections, {
    color: partColors["hands"][1],
    ...globalOpts,
  });

  canvas.restore();
};

const prepareAnim = (ctx: RenderContext, animContext: AnimationContext) => {
  const wordLen = animContext.req.words.length;
  // Wait time between words
  const waitTime = 300;
  const canvasCtx = ctx.canvas.getContext("2d")!;

  // == MAIN ANIMATION LOOP ==
  function mainLoop(now: number) {
    requestAnimationFrame(mainLoop);

    if (animContext.paused) {
      return;
    }

    const currWordPos = animContext.currentWord;
    const currWord = animContext.req.words[currWordPos];
    const currWordData = animContext.req.dataMap[currWord] ?? [];

    const currFrame = animContext.currentFrame;

    if (currFrame >= (currWordData.frames?.length ?? 0)) {
      // Next Word
      animContext.paused = true;
      setTimeout(() => {
        animContext.currentWord = (animContext.currentWord + 1) % wordLen;
        animContext.currentFrame = 0;
        animContext.paused = false;
      }, waitTime);
    } else {
      // Still on this word
      const currFrameData = currWordData?.frames?.[animContext.currentFrame];
      animContext.currentFps = currWordData.fps ?? 30; // Assume 30 FPS

      // 1000 ms in a sec, we want X FPS, so 1000 / X is how long we need to wait.
      const targetInterval = 1000 / animContext.currentFps;

      if (now - animContext.lastRender > targetInterval) {
        animContext.lastRender = now;
        if (currFrame === 0) {
          animContext.wordCallback?.(currWordPos);
          if (currFrameData === undefined) {
            // Our word is unknown, skip it
            animContext.currentWord = (animContext.currentWord + 1) % wordLen;
            return;
          } else {
            renderFrame(ctx, canvasCtx, currFrameData);
          }
        } else {
          renderFrame(ctx, canvasCtx, currFrameData);
        }
        animContext.currentFrame += 1;
      }
    }
  }

  requestAnimationFrame(mainLoop);
};

export const prepareCanvas = (canvas: HTMLCanvasElement): RenderContext => {
  const parent = canvas.parentElement!;
  const width = parent.offsetWidth;
  const height = parent.offsetHeight;

  canvas.setAttribute("width", `${Math.floor(width)}px`);
  canvas.setAttribute("height", `${Math.floor(height)}px`);

  parent.appendChild(canvas);

  const ctx = {
    canvas,
    dimensions: [width, height],
  } as RenderContext;

  const onParentResize = (entries: ResizeObserverEntry[]) => {
    const lastItem = entries[entries.length - 1];
    const { inlineSize: width, blockSize: height } = lastItem.contentBoxSize[0];
    canvas.setAttribute("width", `${Math.floor(width)}px`);
    canvas.setAttribute("height", `${Math.floor(height)}px`);

    ctx.dimensions = [width, height];
  };

  const observer = new ResizeObserver(onParentResize);
  observer.observe(parent);

  return ctx;
};

export const renderAsl = (
  ctx: RenderContext,
  req: TranslationRequest,
  onWordChange?: (index: number) => void,
) => {
  if (ctx.anim) {
    ctx.anim.req = req;
    ctx.anim.paused = false;
    ctx.anim.currentWord = 0;
    ctx.anim.currentFrame = 0;
    ctx.anim.wordCallback = onWordChange;
  } else {
    const animContext = {
      req,
      currentWord: 0,
      currentFrame: 0,
      wordCallback: onWordChange,
      paused: false,
      currentFps: 30,
      lastRender: window.performance.now(),
    } as AnimationContext;
    prepareAnim(ctx, animContext);
  }
};
