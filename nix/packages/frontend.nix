{pkgs, ...}:
pkgs.buildNpmPackage {
  name = "asl-research-frontend";
  version = "0.1.0";
  src = ../../web-frontend;
  packageJSON = ../../web-frontend/package.json;
  npmDeps = pkgs.importNpmLock {
    npmRoot = ../../web-frontend;
  };
  npmConfigHook = pkgs.importNpmLock.npmConfigHook;
  installPhase = "cp -r dist/ $out";
  ASTRO_TELEMETRY_DISABLED = 1;
}
