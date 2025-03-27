{pkgs, ...}:
pkgs.buildNpmPackage (let
  src = ../../web-frontend;
in {
  name = "asl-research-frontend";
  version = "0.1.0";
  inherit src;
  packageJSON = src + "/package.json";
  npmDeps = pkgs.importNpmLock {
    npmRoot = src;
  };
  npmConfigHook = pkgs.importNpmLock.npmConfigHook;
  installPhase = "cp -r dist/ $out";
  ASTRO_TELEMETRY_DISABLED = 1;
})
