/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <basalt/io/dataset_io.h>
#include <basalt/utils/filesystem.h>

#include <opencv2/highgui/highgui.hpp>

namespace basalt {

class KittiVioDataset : public VioDataset {
  size_t num_cams;

  std::string path;

  std::vector<int64_t> image_timestamps;
  std::unordered_map<int64_t, std::string> image_path;

  // vector of images for every timestamp
  // assumes vectors size is num_cams for every timestamp with null pointers for
  // missing frames
  // std::unordered_map<int64_t, std::vector<ImageData>> image_data;

  Eigen::aligned_vector<AccelData> accel_data;
  Eigen::aligned_vector<GyroData> gyro_data;

  std::vector<int64_t> gt_timestamps;  // ordered gt timestamps
  Eigen::aligned_vector<Sophus::SE3d>
      gt_pose_data;  // TODO: change to eigen aligned

  int64_t mocap_to_imu_offset_ns;

 public:
  ~KittiVioDataset(){};

  size_t get_num_cams() const { return num_cams; }

  std::vector<int64_t> &get_image_timestamps() { return image_timestamps; }

  const Eigen::aligned_vector<AccelData> &get_accel_data() const {
    return accel_data;
  }
  const Eigen::aligned_vector<GyroData> &get_gyro_data() const {
    return gyro_data;
  }
  const std::vector<int64_t> &get_gt_timestamps() const {
    return gt_timestamps;
  }
  const Eigen::aligned_vector<Sophus::SE3d> &get_gt_pose_data() const {
    return gt_pose_data;
  }
  int64_t get_mocap_to_imu_offset_ns() const { return mocap_to_imu_offset_ns; }

  std::vector<ImageData> get_image_data(int64_t t_ns) {
    std::vector<ImageData> res(num_cams);

    const std::vector<std::string> folder = {"/image_00/data/", "/image_01/data/"};

    for (size_t i = 0; i < num_cams; i++) {
      std::string full_image_path = path + folder[i] + image_path[t_ns];

      if (fs::exists(full_image_path)) {
        cv::Mat img = cv::imread(full_image_path, cv::IMREAD_UNCHANGED);

        if (img.type() == CV_8UC1) {
          res[i].img.reset(new ManagedImage<uint16_t>(img.cols, img.rows));

          const uint8_t *data_in = img.ptr();
          uint16_t *data_out = res[i].img->ptr;

          size_t full_size = img.cols * img.rows;
          for (size_t i = 0; i < full_size; i++) {
            int val = data_in[i];
            val = val << 8;
            data_out[i] = val;
          }
        } else {
          std::cerr << "img.fmt.bpp " << img.type() << std::endl;
          std::abort();
        }
      }
    }

    return res;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  friend class KittiIO;
};

class KittiIO : public DatasetIoInterface {
 public:
  KittiIO() {}

  void read(const std::string &path) {
    if (!fs::exists(path))
      std::cerr << "No dataset found in " << path << std::endl;

    data.reset(new KittiVioDataset);

    data->num_cams = 2;
    data->path = path;

    read_image_timestamps(path + "/image_00/timestamps.txt");
    read_imu_data(path + "/oxts");

    if (fs::exists(path + "/poses.txt")) {
      read_gt_data_pose(path + "/poses.txt");
    }
  }

  void read_imu_data(const std::string &path) {
    data->accel_data.clear();
    data->gyro_data.clear();

    std::ifstream ts(path + "/100timestamps.txt");    
    std::string line;
    int imuId = 0;

    while (std::getline(ts, line)) {    // imu timestamps
      uint64_t timestamp = timeToTimestamps(line);
      std::stringstream imuDataPath;
      imuDataPath << std::setfill('0') << std::setw(10) << imuId << ".txt";
      std::ifstream imuDataFile(path + "/100data/" + imuDataPath.str());  //100hz imu folder
      std::string imuLine;

      if (std::getline(imuDataFile, imuLine)){
        Eigen::Vector3d gyro, accel;
        char *cha = (char*)imuLine.data();
        double tmpd;
        int tmpi;
        Eigen::Matrix3d kittiImu2FactImu;
        kittiImu2FactImu << -1, 0, 0,
                            0, -1, 0,
                            0, 0, 1;

        sscanf(cha, "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %d %d %d %d %d", 
                           &tmpd, &tmpd, &tmpd, &tmpd, &tmpd, &tmpd, &tmpd, &tmpd, &tmpd, &tmpd,
                           &tmpd, &accel[0], &accel[1], &accel[2], &tmpd, &tmpd, &tmpd, &gyro[0], &gyro[1], &gyro[2],
                           &tmpd, &tmpd, &tmpd, &tmpd, &tmpd, &tmpi, &tmpi, &tmpi, &tmpi, &tmpi);

        data->accel_data.emplace_back();
        data->accel_data.back().timestamp_ns = timestamp;
        data->accel_data.back().data = kittiImu2FactImu * accel;

        data->gyro_data.emplace_back();
        data->gyro_data.back().timestamp_ns = timestamp;
        data->gyro_data.back().data = kittiImu2FactImu * gyro;

        imuId +=1;
      }else{  //imu data less timestamps
        break;
      }
    }
  }

  void reset() { data.reset(); }

  VioDatasetPtr get_data() { return data; }

 private:

 uint64_t timeToTimestamps(std::string line){
    tm tm_;
    int year, month, day, hour, minute, second, nsecond;
    char *c = (char*)line.data();

    sscanf(c, "%d-%d-%d %d:%d:%d.%d", &year, &month, &day, &hour, &minute, &second, &nsecond);
    tm_.tm_year = year - 1900;
    tm_.tm_mon = month - 1;
    tm_.tm_mday = day;
    tm_.tm_hour = hour;
    tm_.tm_min = minute;
    tm_.tm_sec = second;
    time_t t_ = mktime(&tm_);

    return t_*1000000000 + nsecond;;
  }

  //without imu
  void read_image_timestamps(const std::string &path) {
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      int64_t t_ns = timeToTimestamps(line);      
      std::stringstream ss1;
      ss1 << std::setfill('0') << std::setw(10) << data->image_timestamps.size()
          << ".png";

      data->image_timestamps.emplace_back(t_ns);
      data->image_path[t_ns] = ss1.str();
    }
  }

  void read_gt_data_pose(const std::string &path) {
    data->gt_timestamps.clear();
    data->gt_pose_data.clear();

    int i = 0;

    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      Eigen::Matrix3d rot;
      Eigen::Vector3d pos;

      ss >> rot(0, 0) >> rot(0, 1) >> rot(0, 2) >> pos[0] >> rot(1, 0) >>
          rot(1, 1) >> rot(1, 2) >> pos[1] >> rot(2, 0) >> rot(2, 1) >>
          rot(2, 2) >> pos[2];

      data->gt_timestamps.emplace_back(data->image_timestamps[i]);
      data->gt_pose_data.emplace_back(Eigen::Quaterniond(rot), pos);
      i++;
    }
  }

  std::shared_ptr<KittiVioDataset> data;
};

}  // namespace basalt
