{outputs, ...}: let
  # text/html included by default
  compressMimeTypes = [
    "application/atom+xml"
    "application/geo+json"
    "application/javascript"
    "application/json"
    "application/ld+json"
    "application/manifest+json"
    "application/rdf+xml"
    "application/vnd.ms-fontobject"
    "application/wasm"
    "application/x-rss+xml"
    "application/x-web-app-manifest+json"
    "application/xhtml+xml"
    "application/xliff+xml"
    "application/xml"
    "application/x-msgpack"
    "font/collection"
    "font/otf"
    "font/ttf"
    "image/bmp"
    "image/svg+xml"
    "image/vnd.microsoft.icon"
    "text/cache-manifest"
    "text/calendar"
    "text/css"
    "text/csv"
    "text/javascript"
    "text/markdown"
    "text/plain"
    "text/vcard"
    "text/vnd.rim.location.xloc"
    "text/vtt"
    "text/x-component"
    "text/xml"
  ];
in {
  system = "x86_64-linux";

  modules = [
    outputs.nixosModules.aslSite
    (
      {
        pkgs,
        modulesPath,
        lib,
        ...
      }: {
        imports = ["${modulesPath}/virtualisation/qemu-vm.nix"];

        virtualisation = {
          graphics = false;
          diskImage = null;
          sharedDirectories = {
            words = {
              source = "\${WORD_DIR:-$PROJ_ROOT/words}";
              target = "/word-mnt";
            };
            models = {
              source = "\${MODELS_DIR:-$PROJ_ROOT/saved_models}";
              target = "/models-mnt";
            };
            cert = {
              source = "\${CERTS_DIR:-$PROJ_ROOT/certs}";
              target = "/certs-mnt";
            };
          };
          forwardPorts = [
            {
              from = "host";
              guest.port = 8080;
              host.port = 8080;
            }
            {
              from = "host";
              guest.port = 8443;
              host.port = 8443;
            }
          ];
        };

        users = {
          mutableUsers = false;
          users."root".hashedPassword = "$y$j9T$sfyYVDmioHNdul1/euqAQ1$JFPf2l70Nw.rfH2ku7kr5oHZebJoRm9UDiWX4g3v7k9";
        };

        system.stateVersion = "25.05";
        time.timeZone = "America/New_York";

        services.aslSite = {
          enable = true;
          wordDir = "/word-mnt";
          models = {
            englishToGloss = "/models-mnt/english_to_gloss.pt";
            glossToEnglish = "/models-mnt/gloss_to_english.pt";
            aslToGloss = "/models-mnt/asl_to_gloss.pt";
          };
        };

        networking.firewall.allowedTCPPorts = [8080 8443];

        services.nginx = {
          enable = true;
          recommendedOptimisation = true;
          recommendedTlsSettings = true;
          defaultHTTPListenPort = 8080;
          defaultSSLListenPort = 8443;
          # recommendedBrotliSettings doesn't include application/x-msgpack
          # and sadly doesn't expose compressMimeTypes, so we'll set what it
          # does manually
          additionalModules = with pkgs.nginxModules; [brotli];
          appendHttpConfig = ''
            expires max;
            brotli on;
            brotli_static on;
            brotli_comp_level 5;
            brotli_window 512k;
            brotli_min_length 256;
            brotli_types ${lib.concatStringsSep " " compressMimeTypes};
          '';
          virtualHosts.aslResearch = {
            default = true;
            onlySSL = true;
            sslCertificate = "/certs-mnt/host.cert";
            sslCertificateKey = "/certs-mnt/host.key";
            root = "${pkgs.frontend}";
            locations."/api" = {
              recommendedProxySettings = true;
              proxyPass = "http://127.0.0.1:8000";
              # Don't cache api routes
              extraConfig = ''
                expires off;
              '';
            };
          };
        };
      }
    )
  ];
}
