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
    std::string pose_model = "D:/Dev/Project/Test_RTMPose/models/RTMPose/halpe26_rtmpose-l_256x192";
    std::string input = "C:/Capture/0408_Calibration/HSH/motion1/export";
    std::string output = "D:/Dev/Project/Test_RTMPose/demo/outputs";

    std::string skeleton = "halpe26";  // "coco", "coco-wholebody", "halpe26"
    std::string device = "cuda";              // "cuda", "cpu"
    int output_size = 0;                      // "Long-edge of output frames" (0: original size)
    int flip = 0;                             // "Set to 1 for flipping the input horizontally"
    int show = -1;                            // "Delay passed to `cv::waitKey` when using `cv::imshow`; -1: disable"
    std::string background = "default";       // Output background, "default": original image, "black": black background




    double total_preprocess_time = 0.0;
    double total_inference_time = 0.0;
    double total_postprocess_time = 0.0;
    int processed_frames = 0;

    auto total_start_time = std::chrono::high_resolution_clock::now();
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
        std::filesystem::path output_dir = std::filesystem::path(output);
        std::string output_path = (output_dir / output_filename).string();

        double fps = input_streams[i].get_fps();
        output_streams.emplace_back(output_path, show, fps);
    }

    utils::Visualize v(output_size);
    v.set_background(background);
    v.set_skeleton(utils::Skeleton::get(skeleton));

    for (int frame_idx = 0; frame_idx < total_frames; frame_idx++) {
        
        
        
        auto preprocess_start = std::chrono::high_resolution_clock::now();

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


        auto preprocess_end = std::chrono::high_resolution_clock::now();
        total_preprocess_time += std::chrono::duration<double>(preprocess_end - preprocess_start).count();

        auto inference_start = std::chrono::high_resolution_clock::now();



        // Inference
        std::vector<mmdeploy::PoseTracker::Result> batch_results = tracker.Apply(states, mmdeploy_frames);


        auto inference_end = std::chrono::high_resolution_clock::now();
        total_inference_time += std::chrono::duration<double>(inference_end - inference_start).count();
        
        auto postprocess_start = std::chrono::high_resolution_clock::now();


        std::vector<decltype(v.get_session(batch_frames[0]))> sessions;
        sessions.reserve(video_files.size());

        for (size_t i = 0; i < video_files.size(); i++) {
            sessions.emplace_back(v.get_session(batch_frames[i]));
        }

        for (size_t i = 0; i < video_files.size(); i++) {
            for (const mmdeploy_pose_tracker_target_t& target : batch_results[i]) {
                sessions[i].add_pose(target.keypoints, target.scores, target.keypoint_count, FLAGS_pose_kpt_thr);
            }
        }

        for (size_t i = 0; i < video_files.size(); i++) {
            if (!output_streams[i].write(sessions[i].get())) {
                std::cout << "User requested exit for video " << i << std::endl;
                break;
            }
        }

        auto postprocess_end = std::chrono::high_resolution_clock::now();
        total_postprocess_time += std::chrono::duration<double>(postprocess_end - postprocess_start).count();
        processed_frames++;
    }

    auto total_end_time = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end_time - total_start_time).count();

    std::cout << "Batch processing completed!" << std::endl;


    if (processed_frames > 0) {
        double avg_preprocess_fps = processed_frames / total_preprocess_time;
        double avg_inference_fps = processed_frames / total_inference_time;
        double avg_postprocess_fps = processed_frames / total_postprocess_time;
        double avg_total_fps = processed_frames / total_time;

        std::cout << "\n=== Performance Results ===" << std::endl;
        std::cout << "Total Processed Frames: " << processed_frames << std::endl;
        std::cout << "Total Processing Time: " << total_time << " seconds" << std::endl;
        std::cout << "Preprocess Time: " << total_preprocess_time << " seconds (FPS: " << avg_preprocess_fps << ")" << std::endl;
        std::cout << "Inference Time: " << total_inference_time << " seconds (FPS: " << avg_inference_fps << ")" << std::endl;
        std::cout << "Postprocess Time: " << total_postprocess_time << " seconds (FPS: " << avg_postprocess_fps << ")" << std::endl;
        std::cout << "Average Total FPS: " << avg_total_fps << std::endl;
    }

    return 0;
}