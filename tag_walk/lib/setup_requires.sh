#!/bin/bash

install() {
	case "$(uname -s)" in

		Darwin)
			brew install "${1}"
			;;

		Linux)
			apt-get install "${1}"
			;;
	esac
}

linux_install() {
	case "$(uname -s)" in
		Linux)
			apt-get install "${1}"
			;;
	esac
}

install "awscli"
install "postgresql"

linux_install "build-essential libssl-dev libffi-dev python3-dev"

linux_install "python3-pip"

linux_install "libhdf5-10 libhdf5-dev"
linux_install "python3-h5py"
linux_install "python3-numpy"
linux_install "python3-scipy"
linux_install "python3-pandas"

pip3 install -r ./lib/requirements.txt
