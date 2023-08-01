#~/bin/bash
# install for gcc
yum install libaio-devel -y
yum install centos-release-scl -y
yum-config-manager --enable rhel-server-rhscl-7-rpms -y
yum install devtoolset-8 -y
yum install llvm-toolset-7 -y
sudo yum -y install llvm-toolset-7-clang-analyzer llvm-toolset-7-clang-tools-extra
sudo yum -y install pdsh
scl enable devtoolset-8 llvm-toolset-7 bash
