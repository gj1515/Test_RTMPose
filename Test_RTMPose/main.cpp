// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/pose_tracker.hpp"
#include "utils/argparse.h"
#include "utils/mediaio.h"
#include "utils/visualize.h"
#include "pose_tracker_params.h"

#include <filesystem>
#include <algorithm>
#include <chrono>

std::vector<std::string> getVideoFiles(const std::string& folder_path) {
    std::vector<std::string> video_files;

    if (!std::filesystem::exists(folder_path)) {
        std::cerr << "Error: Folder does not exist: " << folder_path << std::endl;
        return video_files;
    }

    if (!std::filesystem::is_directory(folder_path)) {
        std::cerr << "Error: Path is not a directory: " << folder_path << std::endl;
        return video_files;
    }

    for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::string ext = utils::mediaio::detail::get_extension(file_path);
            if (utils::mediaio::detail::is_video(ext)) {
                video_files.push_back(file_path);
            }
        }
    }
    std::sort(video_files.begin(), video_files.end());

    for (const auto& file : video_files) {
        std::cout << "  - " << std::filesystem::path(file).filename().string() << std::endl;
    }

    return video_files;
}

bool validateFrameCounts(const std::vector<std::string>& video_paths, int& total_frames) {
    if (video_paths.empty()) {
        std::cerr << "Error: No video files found." << std::endl;
        return false;
    }

    total_frames = -1;

    for (size_t i = 0; i < video_paths.size(); i++) {
        cv::VideoCapture cap(video_paths[i]);
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video file: " << video_paths[i] << std::endl;
            return false;
        }

        int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

        if (total_frames == -1) {
            total_frames = frame_count;
        }
        else if (total_frames != frame_count) {
            std::cerr << "Error: All videos must have the same frame count!" << std::endl;
            std::cerr << "Expected frame count: " << total_frames << ", Current video frame count: " << frame_count << std::endl;
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::string det_model = "D:/Dev/Project/Test_RTMPose/models/RTMDet/rtmdet-n_320x320";
    std::string pose_model = "D:/Dev/Project/Test_RTMPose/models/RTMPose/rtmpose-m_halpe26_256x192";
    std::string input = "C:/Capture/0408_Calibration/HSH/motion1/export";
    std::string output = "D:/Dev/Project/Test_RTMPose/demo/outputs/output.mp4";

    std::string skeleton = "halpe26";
    std::string device = "cuda";
    int output_size = 0;
    int flip = 0;
    int show = -1;
    std::string background = "default";

    std::vector<std::string> video_files = getVideoFiles(input);

    if (video_files.empty()) {
        std::cerr << "Error: No video files to process." << std::endl;
        return -1;
    }

    int total_frames;
    if (!validateFrameCounts(video_files, total_frames)) {
        return -1;
    }

    // create pose tracker pipeline
    mmdeploy::PoseTracker tracker(mmdeploy::Model(det_model), mmdeploy::Model(pose_model), mmdeploy::Device{ device });

    mmdeploy::PoseTracker::Params params;
    InitTrackerParams(params);

    std::vector<mmdeploy::PoseTracker::State> states;
    std::vector<utils::mediaio::Input> input_streams;
    std::vector<utils::mediaio::Output> output_streams;

    for (size_t i = 0; i < video_files.size(); i++) {
        // create a tracker state for each video
        states.emplace_back(tracker.CreateState(params));

        input_streams.emplace_back(video_files[i], flip);

        std::filesystem::path input_path(video_files[i]);
        std::string output_filename = input_path.filename().string();
        std::filesystem::path output_dir = std::filesystem::path(output).parent_path();
        std::string output_path = (output_dir / output_filename).string();

        double fps = input_streams[i].get_fps();
        output_streams.emplace_back(output_path, show, fps);
    }

    utils::Visualize v(output_size);
    v.set_background(background);
    v.set_skeleton(utils::Skeleton::get(skeleton));

    // FPS measurement variables
    auto total_start_time = std::chrono::high_resolution_clock::now();
    auto total_inference_time = std::chrono::duration<double>::zero();
    int processed_frames = 0;

    for (int frame_idx = 0; frame_idx < total_frames; frame_idx++) {
        std::vector<cv::Mat> batch_frames;
        batch_frames.reserve(video_files.size());

        bool all_frames_valid = true;
        for (size_t i = 0; i < video_files.size(); i++) {
            cv::Mat frame = input_streams[i].read();
            if (frame.empty()) {
                std::cerr << "Error: Failed to read frame " << frame_idx
                    << " from video " << i << std::endl;
                all_frames_valid = false;
                break;
            }
            batch_frames.push_back(frame);
        }

        if (!all_frames_valid) {
            break;
        }

        // cv::Mat to mmdeploy::Mat
        std::vector<mmdeploy::Mat> mmdeploy_frames;
        mmdeploy_frames.reserve(batch_frames.size());
        for (const auto& frame : batch_frames) {
            mmdeploy_frames.push_back(mmdeploy::Mat(frame));
        }

        // Measure inference time
        auto inference_start = std::chrono::high_resolution_clock::now();

        // Inference
        std::vector<mmdeploy::PoseTracker::Result> batch_results = tracker.Apply(states, mmdeploy_frames);

        auto inference_end = std::chrono::high_resolution_clock::now();
        total_inference_time += std::chrono::duration<double>(inference_end - inference_start);

        for (size_t i = 0; i < video_files.size(); i++) {
            auto sess = v.get_session(batch_frames[i]);

            for (const mmdeploy_pose_tracker_target_t& target : batch_results[i]) {
                sess.add_pose(target.keypoints, target.scores, target.keypoint_count, FLAGS_pose_kpt_thr);
            }

            if (!output_streams[i].write(sess.get())) {
                std::cout << "User requested exit for video " << i << std::endl;
                break;
            }
        }

        processed_frames++;

        if (frame_idx % 100 == 0) {
            std::cout << "Processed " << frame_idx << "/" << total_frames << " frames" << std::endl;
        }
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    auto total_processing_time = std::chrono::duration<double>(total_end_time - total_start_time);

    std::cout << "Batch processing completed!" << std::endl;

    // Calculate and display FPS results
    if (processed_frames > 0) {
        double total_fps = processed_frames / total_processing_time.count();
        double inference_fps = processed_frames / total_inference_time.count();

        std::cout << "=== Performance Results ===" << std::endl;
        std::cout << "Total processed frames: " << processed_frames << std::endl;
        std::cout << "Total processing time: " << total_processing_time.count() << " seconds" << std::endl;
        std::cout << "Total inference time: " << total_inference_time.count() << " seconds" << std::endl;
        std::cout << "Overall FPS: " << total_fps << std::endl;
        std::cout << "Inference FPS: " << inference_fps << std::endl;
    }

    return 0;
}