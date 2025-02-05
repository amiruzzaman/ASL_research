{
  python3Packages,
  opencv,
  bazel,
  fetchPypi,
  autoPatchelfHook,
  protobuf,
  ...
}:
python3Packages.buildPythonPackage rec {
  pname = "mediapipe";
  version = "0.10.20";
  format = "wheel";

  src = fetchPypi {
    inherit pname version format;
    dist = "cp312";
    python = "cp312";
    abi = "cp312";

    platform = "manylinux_2_28_x86_64";

    sha256 = "sha256-8EUrfIY+wM0cY2DwuB/iSSxWCI3oCt+PNkLFe0Xh8dM=";
  };

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
