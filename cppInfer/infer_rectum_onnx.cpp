#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <onnxruntime/onnxruntime_cxx_api.h>

#include "cnpy/cnpy.h"

namespace fs = std::filesystem;

namespace {

constexpr int kDefaultImageSize = 224;
constexpr int kPointsPerClass = 10;
constexpr int kClassCount = 2;

struct Args {
    fs::path model_path;
    std::string mode;
    fs::path input_dir;
    fs::path output_dir;
    int img_size = kDefaultImageSize;
};

struct SliceData {
    std::vector<double> image;
    std::vector<double> label;
    size_t height = 0;
    size_t width = 0;
};

struct PointPrompt {
    std::vector<float> points;
    std::vector<int64_t> point_labels;
};

template <typename T>
std::vector<T> to_row_major(const cnpy::NpyArray& array) {
    if (array.shape.size() != 2) {
        throw std::runtime_error("Expected a 2D array in NPZ input");
    }

    const size_t height = array.shape[0];
    const size_t width = array.shape[1];
    std::vector<T> output(height * width);

    auto read_value = [&](size_t index) -> T {
        if (array.word_size == sizeof(double)) {
            return static_cast<T>(array.data<double>()[index]);
        }
        if (array.word_size == sizeof(float)) {
            return static_cast<T>(array.data<float>()[index]);
        }
        if (array.word_size == sizeof(uint8_t)) {
            return static_cast<T>(array.data<uint8_t>()[index]);
        }
        if (array.word_size == sizeof(int64_t)) {
            return static_cast<T>(array.data<int64_t>()[index]);
        }
        throw std::runtime_error("Unsupported array element width in NPZ input");
    };

    if (array.fortran_order) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                output[y * width + x] = read_value(y + height * x);
            }
        }
    } else {
        for (size_t index = 0; index < height * width; ++index) {
            output[index] = read_value(index);
        }
    }
    return output;
}

std::vector<double> to_fortran_major(const std::vector<double>& row_major, size_t height, size_t width) {
    std::vector<double> output(height * width);
    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            output[y + height * x] = row_major[y * width + x];
        }
    }
    return output;
}

Args parse_args(int argc, char** argv) {
    Args args;

    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        auto require_value = [&](const std::string& option) -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error("Missing value for option: " + option);
            }
            return argv[++index];
        };

        if (key == "--model") {
            args.model_path = require_value(key);
        } else if (key == "--mode") {
            args.mode = require_value(key);
        } else if (key == "--input") {
            args.input_dir = require_value(key);
        } else if (key == "--output") {
            args.output_dir = require_value(key);
        } else if (key == "--img_size") {
            args.img_size = std::stoi(require_value(key));
        } else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: infer_rectum_onnx --model <path> --mode <mode> --input <dir> --output <dir> [--img_size 224]\n"
                << "Modes: no_prompt, box, pts, box+pts, sota\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown option: " + key);
        }
    }

    const std::vector<std::string> valid_modes = {"no_prompt", "box", "pts", "box+pts", "sota"};
    if (args.model_path.empty() || args.mode.empty() || args.input_dir.empty() || args.output_dir.empty()) {
        throw std::runtime_error("--model, --mode, --input and --output are required");
    }
    if (std::find(valid_modes.begin(), valid_modes.end(), args.mode) == valid_modes.end()) {
        throw std::runtime_error("Unsupported mode: " + args.mode);
    }
    return args;
}

std::vector<fs::path> sorted_npz_files(const fs::path& input_dir) {
    if (!fs::exists(input_dir)) {
        throw std::runtime_error("Input directory does not exist: " + input_dir.string());
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".npz") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) {
        throw std::runtime_error("No NPZ files found in input directory: " + input_dir.string());
    }
    return files;
}

SliceData load_slice(const fs::path& path) {
    const cnpy::NpyArray image_array = cnpy::npz_load(path.string(), "image");
    const cnpy::NpyArray label_array = cnpy::npz_load(path.string(), "label");
    if (image_array.shape != label_array.shape) {
        throw std::runtime_error("image and label shapes do not match in " + path.string());
    }

    SliceData slice;
    slice.height = image_array.shape[0];
    slice.width = image_array.shape[1];
    slice.image = to_row_major<double>(image_array);
    slice.label = to_row_major<double>(label_array);
    return slice;
}

std::vector<float> resize_bilinear(const std::vector<double>& image, size_t input_h, size_t input_w, int output_h, int output_w) {
    std::vector<float> output(static_cast<size_t>(output_h) * static_cast<size_t>(output_w), 0.0f);
    const double scale_y = static_cast<double>(input_h) / static_cast<double>(output_h);
    const double scale_x = static_cast<double>(input_w) / static_cast<double>(output_w);

    for (int out_y = 0; out_y < output_h; ++out_y) {
        const double source_y = (static_cast<double>(out_y) + 0.5) * scale_y - 0.5;
        const int y0 = std::max(0, static_cast<int>(std::floor(source_y)));
        const int y1 = std::min(static_cast<int>(input_h) - 1, y0 + 1);
        const double ly = source_y - static_cast<double>(y0);
        const double hy = 1.0 - ly;

        for (int out_x = 0; out_x < output_w; ++out_x) {
            const double source_x = (static_cast<double>(out_x) + 0.5) * scale_x - 0.5;
            const int x0 = std::max(0, static_cast<int>(std::floor(source_x)));
            const int x1 = std::min(static_cast<int>(input_w) - 1, x0 + 1);
            const double lx = source_x - static_cast<double>(x0);
            const double hx = 1.0 - lx;

            const double top = image[static_cast<size_t>(y0) * input_w + static_cast<size_t>(x0)] * hx
                + image[static_cast<size_t>(y0) * input_w + static_cast<size_t>(x1)] * lx;
            const double bottom = image[static_cast<size_t>(y1) * input_w + static_cast<size_t>(x0)] * hx
                + image[static_cast<size_t>(y1) * input_w + static_cast<size_t>(x1)] * lx;
            output[static_cast<size_t>(out_y) * static_cast<size_t>(output_w) + static_cast<size_t>(out_x)] =
                static_cast<float>(top * hy + bottom * ly);
        }
    }
    return output;
}

std::vector<int64_t> resize_mask_nearest(const std::vector<double>& mask, size_t input_h, size_t input_w, int output_h, int output_w) {
    std::vector<int64_t> output(static_cast<size_t>(output_h) * static_cast<size_t>(output_w), 0);
    for (int out_y = 0; out_y < output_h; ++out_y) {
        const size_t source_y = std::min(
            static_cast<size_t>(std::floor(static_cast<double>(out_y) * static_cast<double>(input_h) / output_h)),
            input_h - 1
        );
        for (int out_x = 0; out_x < output_w; ++out_x) {
            const size_t source_x = std::min(
                static_cast<size_t>(std::floor(static_cast<double>(out_x) * static_cast<double>(input_w) / output_w)),
                input_w - 1
            );
            output[static_cast<size_t>(out_y) * static_cast<size_t>(output_w) + static_cast<size_t>(out_x)] =
                static_cast<int64_t>(std::llround(mask[source_y * input_w + source_x]));
        }
    }
    return output;
}

std::vector<double> resize_mask_nearest_double(const std::vector<double>& mask, size_t input_h, size_t input_w, size_t output_h, size_t output_w) {
    std::vector<double> output(output_h * output_w, 0.0);
    for (size_t out_y = 0; out_y < output_h; ++out_y) {
        const size_t source_y = std::min(static_cast<size_t>(std::floor(static_cast<double>(out_y) * input_h / output_h)), input_h - 1);
        for (size_t out_x = 0; out_x < output_w; ++out_x) {
            const size_t source_x = std::min(static_cast<size_t>(std::floor(static_cast<double>(out_x) * input_w / output_w)), input_w - 1);
            output[out_y * output_w + out_x] = mask[source_y * input_w + source_x];
        }
    }
    return output;
}

std::vector<float> build_image_tensor(
    const std::vector<fs::path>& files,
    size_t index,
    const std::string& mode,
    int img_size,
    const SliceData& current
) {
    std::vector<SliceData> channels;
    channels.reserve(3);

    if (mode == "sota") {
        channels.push_back(load_slice(files[index == 0 ? 0 : index - 1]));
        channels.push_back(current);
        channels.push_back(load_slice(files[index + 1 >= files.size() ? files.size() - 1 : index + 1]));
    } else {
        channels = {current, current, current};
    }

    std::vector<float> tensor(static_cast<size_t>(3) * img_size * img_size);
    for (size_t channel = 0; channel < channels.size(); ++channel) {
        const std::vector<float> resized = resize_bilinear(channels[channel].image, current.height, current.width, img_size, img_size);
        std::copy(
            resized.begin(),
            resized.end(),
            tensor.begin() + static_cast<std::ptrdiff_t>(channel * static_cast<size_t>(img_size) * static_cast<size_t>(img_size))
        );
    }
    return tensor;
}

std::vector<float> build_box_prompt(const std::vector<double>& label, size_t height, size_t width, int img_size) {
    size_t min_x = width;
    size_t max_x = 0;
    size_t min_y = height;
    size_t max_y = 0;
    bool found = false;

    for (size_t y = 0; y < height; ++y) {
        for (size_t x = 0; x < width; ++x) {
            if (label[y * width + x] > 0.0) {
                found = true;
                min_x = std::min(min_x, x);
                max_x = std::max(max_x, x);
                min_y = std::min(min_y, y);
                max_y = std::max(max_y, y);
            }
        }
    }

    if (!found) {
        const float center = static_cast<float>(img_size / 2);
        return {center - 1.0f, center - 1.0f, center + 1.0f, center + 1.0f};
    }

    constexpr size_t margin = 2;
    min_x = (min_x > margin) ? min_x - margin : 0;
    min_y = (min_y > margin) ? min_y - margin : 0;
    max_x = std::min(width, max_x + margin);
    max_y = std::min(height, max_y + margin);

    return {
        static_cast<float>(static_cast<double>(min_x) * img_size / width),
        static_cast<float>(static_cast<double>(min_y) * img_size / height),
        static_cast<float>(static_cast<double>(max_x) * img_size / width),
        static_cast<float>(static_cast<double>(max_y) * img_size / height),
    };
}

std::vector<std::array<float, 2>> sample_class_points(const std::vector<int64_t>& mask, int img_size, int cls_idx) {
    std::vector<std::array<float, 2>> locations;
    locations.reserve(static_cast<size_t>(img_size) * static_cast<size_t>(img_size) / 8);
    for (int y = 0; y < img_size; ++y) {
        for (int x = 0; x < img_size; ++x) {
            if (mask[static_cast<size_t>(y) * static_cast<size_t>(img_size) + static_cast<size_t>(x)] == cls_idx) {
                locations.push_back({static_cast<float>(y), static_cast<float>(x)});
            }
        }
    }

    if (locations.empty()) {
        return {};
    }

    const size_t step = std::max(locations.size() / static_cast<size_t>(kPointsPerClass), static_cast<size_t>(1));
    std::vector<std::array<float, 2>> sampled;
    sampled.reserve(kPointsPerClass);
    for (int index = 0; index < kPointsPerClass; ++index) {
        const size_t sample_index = std::min(static_cast<size_t>(index) * step, locations.size() - 1);
        sampled.push_back(locations[sample_index]);
    }
    return sampled;
}

PointPrompt build_point_prompt(const std::vector<double>& label, size_t height, size_t width, int img_size, int expected_count) {
    const std::vector<int64_t> resized_mask = resize_mask_nearest(label, height, width, img_size, img_size);

    std::vector<std::vector<std::array<float, 2>>> groups;
    groups.reserve(kClassCount);
    for (int cls_idx = 1; cls_idx <= kClassCount; ++cls_idx) {
        auto sampled = sample_class_points(resized_mask, img_size, cls_idx);
        if (!sampled.empty()) {
            groups.push_back(std::move(sampled));
        }
    }

    if (groups.empty()) {
        const std::array<float, 2> center = {static_cast<float>(img_size / 2), static_cast<float>(img_size / 2)};
        groups = {
            std::vector<std::array<float, 2>>(kPointsPerClass, center),
            std::vector<std::array<float, 2>>(kPointsPerClass, center),
        };
    } else if (groups.size() == 1) {
        groups.push_back(groups.front());
    }

    PointPrompt prompt;
    for (const auto& group : groups) {
        for (const int selected_index : {0, 4, 8}) {
            const auto& point = group[static_cast<size_t>(selected_index)];
            prompt.points.push_back(point[0]);
            prompt.points.push_back(point[1]);
            prompt.point_labels.push_back(1);
        }
    }

    if (expected_count == 7) {
        prompt.points.push_back(0.0f);
        prompt.points.push_back(0.0f);
        prompt.point_labels.push_back(-1);
    } else if (expected_count < static_cast<int>(prompt.point_labels.size())) {
        prompt.points.resize(static_cast<size_t>(expected_count) * 2);
        prompt.point_labels.resize(static_cast<size_t>(expected_count));
    } else {
        while (static_cast<int>(prompt.point_labels.size()) < expected_count) {
            const float last_y = prompt.points[prompt.points.size() - 2];
            const float last_x = prompt.points[prompt.points.size() - 1];
            prompt.points.push_back(last_y);
            prompt.points.push_back(last_x);
            prompt.point_labels.push_back(1);
        }
    }
    return prompt;
}

std::vector<double> postprocess_prediction(
    const float* logits,
    const std::vector<int64_t>& shape,
    size_t output_height,
    size_t output_width
) {
    if (shape.size() < 3) {
        throw std::runtime_error("Unexpected ONNX output rank");
    }

    size_t classes = 0;
    size_t input_h = 0;
    size_t input_w = 0;
    if (shape.size() == 4) {
        classes = static_cast<size_t>(shape[1]);
        input_h = static_cast<size_t>(shape[2]);
        input_w = static_cast<size_t>(shape[3]);
    } else if (shape.size() == 3) {
        classes = 1;
        input_h = static_cast<size_t>(shape[1]);
        input_w = static_cast<size_t>(shape[2]);
    } else {
        throw std::runtime_error("Unsupported ONNX output shape");
    }

    std::vector<double> prediction(input_h * input_w, 0.0);
    if (classes == 1) {
        for (size_t index = 0; index < input_h * input_w; ++index) {
            prediction[index] = static_cast<double>(logits[index]);
        }
    } else {
        for (size_t y = 0; y < input_h; ++y) {
            for (size_t x = 0; x < input_w; ++x) {
                float best_value = -std::numeric_limits<float>::infinity();
                size_t best_class = 0;
                for (size_t cls = 0; cls < classes; ++cls) {
                    const size_t offset = cls * input_h * input_w + y * input_w + x;
                    if (logits[offset] > best_value) {
                        best_value = logits[offset];
                        best_class = cls;
                    }
                }
                prediction[y * input_w + x] = static_cast<double>(best_class);
            }
        }
    }

    return resize_mask_nearest_double(prediction, input_h, input_w, output_height, output_width);
}

void save_output_npz(const fs::path& path, const SliceData& input_slice, const std::vector<double>& prediction) {
    fs::create_directories(path.parent_path());
    const std::vector<double> image_fortran = to_fortran_major(input_slice.image, input_slice.height, input_slice.width);
    const std::vector<double> label_fortran = to_fortran_major(prediction, input_slice.height, input_slice.width);
    const std::vector<size_t> shape = {input_slice.height, input_slice.width};
    cnpy::npz_save(path.string(), "image", image_fortran.data(), shape, "w", true);
    cnpy::npz_save(path.string(), "label", label_fortran.data(), shape, "a", true);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const std::vector<fs::path> input_files = sorted_npz_files(args.input_dir);
        fs::create_directories(args.output_dir);

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "rectum_onnx_cpp");
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetIntraOpNumThreads(1);

        Ort::Session session(env, args.model_path.string().c_str(), session_options);
        Ort::AllocatorWithDefaultOptions allocator;

        std::vector<std::string> input_name_storage;
        std::vector<const char*> input_names;
        input_name_storage.reserve(session.GetInputCount());
        input_names.reserve(session.GetInputCount());

        int point_count = 6;
        for (size_t index = 0; index < session.GetInputCount(); ++index) {
            auto name_ptr = session.GetInputNameAllocated(index, allocator);
            input_name_storage.emplace_back(name_ptr.get());
            input_names.push_back(input_name_storage.back().c_str());

            if (input_name_storage.back() == "points") {
                const auto shape = session.GetInputTypeInfo(index).GetTensorTypeAndShapeInfo().GetShape();
                if (shape.size() > 1 && shape[1] > 0) {
                    point_count = static_cast<int>(shape[1]);
                }
            }
        }

        std::vector<std::string> output_name_storage;
        std::vector<const char*> output_names;
        output_name_storage.reserve(session.GetOutputCount());
        output_names.reserve(session.GetOutputCount());
        for (size_t index = 0; index < session.GetOutputCount(); ++index) {
            auto name_ptr = session.GetOutputNameAllocated(index, allocator);
            output_name_storage.emplace_back(name_ptr.get());
            output_names.push_back(output_name_storage.back().c_str());
        }

        const std::array<int64_t, 4> image_shape = {1, 3, args.img_size, args.img_size};
        const std::array<int64_t, 2> box_shape = {1, 4};
        const std::array<int64_t, 3> point_shape = {1, point_count, 2};
        const std::array<int64_t, 2> point_label_shape = {1, point_count};

        for (size_t file_index = 0; file_index < input_files.size(); ++file_index) {
            const SliceData current = load_slice(input_files[file_index]);
            std::vector<float> image_tensor = build_image_tensor(input_files, file_index, args.mode, args.img_size, current);
            std::vector<float> boxes = build_box_prompt(current.label, current.height, current.width, args.img_size);
            PointPrompt prompt = build_point_prompt(current.label, current.height, current.width, args.img_size, point_count);

            std::vector<Ort::Value> inputs;
            inputs.reserve(input_names.size());

            for (const std::string& input_name : input_name_storage) {
                Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
                if (input_name == "image" || input_name == "input") {
                    inputs.emplace_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        image_tensor.data(),
                        image_tensor.size(),
                        image_shape.data(),
                        image_shape.size()
                    ));
                } else if (input_name == "boxes") {
                    inputs.emplace_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        boxes.data(),
                        boxes.size(),
                        box_shape.data(),
                        box_shape.size()
                    ));
                } else if (input_name == "points") {
                    inputs.emplace_back(Ort::Value::CreateTensor<float>(
                        memory_info,
                        prompt.points.data(),
                        prompt.points.size(),
                        point_shape.data(),
                        point_shape.size()
                    ));
                } else if (input_name == "point_labels" || input_name == "labels") {
                    inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
                        memory_info,
                        prompt.point_labels.data(),
                        prompt.point_labels.size(),
                        point_label_shape.data(),
                        point_label_shape.size()
                    ));
                } else {
                    throw std::runtime_error("Unsupported ONNX input name: " + input_name);
                }
            }

            auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(), output_names.data(), output_names.size());
            if (outputs.empty()) {
                throw std::runtime_error("ONNX Runtime returned no outputs");
            }

            const auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
            const float* logits = outputs[0].GetTensorData<float>();
            const std::vector<double> prediction = postprocess_prediction(logits, output_shape, current.height, current.width);

            save_output_npz(args.output_dir / input_files[file_index].filename(), current, prediction);
        }

        std::cout << "Saved " << input_files.size() << " NPZ files to " << args.output_dir << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << std::endl;
        return 1;
    }
}