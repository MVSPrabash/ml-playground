
#include "../src/Model.hpp"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int testsPassed = 0;
int testsFailed = 0;

void check(bool condition, const std::string& testName) {
    if (condition) {
        ++testsPassed;
        std::cout << "[PASS] " << testName << '\n';
    } else {
        ++testsFailed;
        std::cerr << "[FAIL] " << testName << '\n';
    }
}

template <typename Exception, typename Callable>
void checkThrows(Callable&& callable, const std::string& testName) {
    try {
        callable();
        check(false, testName);
    } catch (const Exception&) {
        check(true, testName);
    } catch (...) {
        check(false, testName + " (unexpected exception type)");
    }
}

// Verify that a matrix has the expected dimensions.
bool hasShape(
    const Math::Matrix& matrix,
    std::size_t rows,
    std::size_t cols
) {
    if (matrix.size() != rows) {
        return false;
    }

    for (const Math::Vector& row : matrix) {
        if (row.size() != cols) {
            return false;
        }
    }

    return true;
}

void testLayerCount() {
    Model model(3, {
        {4, Activation::ReLU, false},
        {2, Activation::Sigmoid, false}
    });

    check(model.nLayers() == 2, "Model reports correct layer count");
    check(model.layers().size() == 2, "layers() returns all layers");
}

void testSingleSampleForward() {
    Model model(3, {
        {4, Activation::ReLU, false},
        {2, Activation::Sigmoid, false}
    });

    const Math::Matrix x = {
        {1.0, 2.0, 3.0}
    };

    const Math::Matrix prediction = model.forward(x);

    check(
        hasShape(prediction, 1, 2),
        "Forward propagates a single sample with correct output shape"
    );
}

void testBatchForward() {
    Model model(3, {
        {4, Activation::ReLU, false},
        {2, Activation::Sigmoid, false}
    });

    const Math::Matrix x = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };

    const Math::Matrix prediction = model.forward(x);

    check(
        hasShape(prediction, 3, 2),
        "Forward preserves batch size and produces correct output dimensions"
    );
}

void testEmptyBatch() {
    Model model(3, {
        {4, Activation::ReLU, false},
        {2, Activation::Sigmoid, false}
    });

    const Math::Matrix x;
    const Math::Matrix prediction = model.forward(x);

    check(
        prediction.empty(),
        "Forward returns an empty matrix for an empty batch"
    );
}

void testInputDimensionMismatch() {
    Model model(3, {
        {4, Activation::ReLU, false},
        {2, Activation::Sigmoid, false}
    });

    const Math::Matrix x = {
        {1.0, 2.0}  // Expected 3 features.
    };

    checkThrows<std::invalid_argument>(
        [&] { (void)model.forward(x); },
        "Forward rejects an incorrect input feature count"
    );
}

void testMalformedBatch() {
    Model model(3, {
        {4, Activation::ReLU, false},
        {2, Activation::Sigmoid, false}
    });

    const Math::Matrix x = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0}  // Malformed second sample.
    };

    checkThrows<std::invalid_argument>(
        [&] { (void)model.forward(x); },
        "Forward rejects a malformed row anywhere in the batch"
    );
}

void testNoLayers() {
    Model model(3, {});

    const Math::Matrix x = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    const Math::Matrix prediction = model.forward(x);

    check(model.nLayers() == 0, "Empty configuration creates zero layers");

    check(
        prediction == x,
        "A model with no layers returns its input unchanged"
    );
}

} // namespace

int main() {
    testLayerCount();
    testSingleSampleForward();
    testBatchForward();
    testEmptyBatch();
    testInputDimensionMismatch();
    testMalformedBatch();
    testNoLayers();

    std::cout << "\nResults: "
              << testsPassed << " passed, "
              << testsFailed << " failed.\n";

    return testsFailed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
