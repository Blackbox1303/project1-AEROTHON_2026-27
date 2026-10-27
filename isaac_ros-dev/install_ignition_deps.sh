#!/bin/bash
# Auto-installs Gazebo Harmonic + MAVROS packages on every new container.
# Sourced by workspace-entrypoint.sh as root before gosu hands off to admin.

# Use mounted workspace dir so marker survives container restarts
MARKER="/workspaces/isaac_ros-dev/.gz_harmonic_deps_installed"

if [[ ! -f "$MARKER" ]]; then
    echo "[entrypoint] Installing Gazebo Harmonic + MAVROS packages..."
    apt-get update -qq

    # Add Gazebo Harmonic (OSRF) apt repo if not already present
    if ! apt-cache show gz-harmonic &>/dev/null 2>&1; then
        echo "[entrypoint] Adding OSRF Gazebo Harmonic apt repo..."
        apt-get install -y --no-install-recommends curl lsb-release gnupg ca-certificates
        curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
            -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] \
http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
            | tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
        apt-get update -qq
    fi

    echo "[entrypoint] Installing gz-harmonic dev packages..."
    # Individual gz dev libs so cmake can find gz-math7Config.cmake, gz-transport13Config.cmake, etc.
    apt-get install -y --no-install-recommends \
        gz-harmonic \
        libgz-math7-dev \
        libgz-transport13-dev \
        libgz-msgs10-dev \
        libgz-sim8-dev \
        libgz-plugin2-dev \
        libgz-rendering8-dev

    echo "[entrypoint] Installing ros-humble-ros-gzharmonic (prebuilt ROS-Gz bridge)..."
    apt-get install -y --no-install-recommends \
        ros-humble-ros-gzharmonic

    echo "[entrypoint] Installing MAVROS..."
    apt-get install -y --no-install-recommends \
        ros-humble-mavros \
        ros-humble-mavros-extras \
        geographiclib-tools

    # Geoid data required by MAVROS/GeographicLib (ignore failure if offline)
    geographiclib-get-geoids egm96-5 || true

    # Install magic_enum header-only library (required by isaac_ros_nitros)
    # Not available as a distro package; install from GitHub release
    echo "[entrypoint] Installing magic_enum..."
    MAGIC_ENUM_VERSION="0.9.5"
    curl -sSL "https://github.com/Neargye/magic_enum/archive/refs/tags/v${MAGIC_ENUM_VERSION}.tar.gz" \
        | tar -xz -C /tmp
    cmake -S /tmp/magic_enum-${MAGIC_ENUM_VERSION} \
          -B /tmp/magic_enum-build \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DMAGIC_ENUM_OPT_BUILD_TESTS=OFF \
          -DMAGIC_ENUM_OPT_BUILD_EXAMPLES=OFF \
          -DMAGIC_ENUM_OPT_INSTALL=ON
    cmake --build /tmp/magic_enum-build --target install
    rm -rf /tmp/magic_enum-${MAGIC_ENUM_VERSION} /tmp/magic_enum-build

    # Run rosdep install using cached DB (skip network YAML + known gz/magic_enum keys)
    echo "[entrypoint] Running rosdep install for workspace packages..."
    rosdep install \
        --from-paths /workspaces/isaac_ros-dev/src \
        --ignore-src \
        -r -y \
        --rosdistro humble \
        --skip-keys "magic_enum gz-harmonic libgz-math7-dev libgz-transport13-dev libgz-msgs10-dev libgz-sim8-dev" \
        || true

    touch "$MARKER"
    echo "[entrypoint] Done – all deps installed."
else
    echo "[entrypoint] Deps already installed, skipping."
fi

# MAVROS: the marker above lives in the MOUNTED workspace, the packages in the --rm container - after the first
# container the block above was skipped and every new container came up without MAVROS (mavros_msgs missing: the
# mission supervisor can't start; 2026-10-24). So check the package itself, every container start (~1 min to install)
if ! dpkg -s ros-humble-mavros-msgs >/dev/null 2>&1 || ! dpkg -s ros-humble-mavros >/dev/null 2>&1; then
    echo "[entrypoint] Installing MAVROS (not in this container)..."
    apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
        ros-humble-mavros ros-humble-mavros-msgs \
        && echo "[entrypoint] MAVROS installed" || echo "[entrypoint] MAVROS install FAILED (offline?)"
fi

# Always export GZ_VERSION so CMakeLists.txt picks the Harmonic branch (not Fortress)
export GZ_VERSION=harmonic
# Persist across all processes launched from this entrypoint
echo "GZ_VERSION=harmonic" >> /etc/environment
