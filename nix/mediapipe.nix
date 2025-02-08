{
  python3Packages,
  opencv,
  bazel,
  fetchPypi,
  autoPatchelfHook,
  protobuf,
  stdenv,
  ...
}:
python3Packages.buildPythonPackage rec {
  pname = "mediapipe";
  version = "0.10.21";
  format = "wheel";

  src = fetchPypi (let
    platformSrc = {
      "x86_64-linux" = {
        platform = "manylinux_2_28_x86_64";
        sha256 = "sha256-lW6x68J1xinmGwhbLKuJw6W56TutG7EHNI2Y2vtaS7U=";
      };
      "x86_64-darwin" = {
        platform = "macosx_11_0_x86_64";
        sha256 = "sha256-lr8Nr6mFHHSx+TAJAZPyPCV8z0uxveu9LKOgI5cBHA4=";
      };
      "aarch64-darwin" = {
        platform = "macosx_11_0_universal2";
        sha256 = "sha256-plUI+MDiinP1GcinF/lZSbnVsvs79KYysYHNtxIk1i4=";
      };
    };
  in
    {
      inherit pname version format;
      dist = "cp312";
      python = "cp312";
      abi = "cp312";
    }
    // (builtins.getAttr stdenv.hostPlatform.system platformSrc));

  dependencies = with python3Packages; [
    absl-py
    attrs
    flatbuffers
    jax
    jaxlib
    matplotlib
    numpy
    opencv-python
    protobuf4
    sounddevice
    sentencepiece
  ];

  nativeBuildInputs = [
    autoPatchelfHook
  ];

  propagatedBuildInputs = [
    opencv
    protobuf
    bazel
  ];
}
