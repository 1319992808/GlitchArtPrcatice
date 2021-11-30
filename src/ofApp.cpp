#include "ofApp.h"

//Multi-threading. The bottleneck of this program is iterating each pixel per frame,
//which could be highly optimized by software concurrency.
static uint64 countPerThread;
static uint64 numThreads = std::thread::hardware_concurrency();

//Declaration of processing funtions
#define DECLARE_POSTPROCESS(func) void func(Mat&, const Mat&, float)
DECLARE_POSTPROCESS(splitRGB1);
DECLARE_POSTPROCESS(splitRGB2);
DECLARE_POSTPROCESS(block1);
DECLARE_POSTPROCESS(block2);
DECLARE_POSTPROCESS(sand);
DECLARE_POSTPROCESS(scanLine); 
DECLARE_POSTPROCESS(digitalStripe);
DECLARE_POSTPROCESS(intDigitalStripe);

//--------------------------------------------------------------
void ofApp::setup(){
	//The image merged in the program.
	img.load("cyber.png");
	img.setImageType(OF_IMAGE_COLOR);
	matImg = toCv(img);
	//Multi-threading context determined at run time.
	countPerThread = (matImg.cols * matImg.rows) / numThreads;
	//Result show on screen
	matResult = Mat(matImg.rows, matImg.cols, CV_8UC3);

	videoGrabber.setup(img.getWidth(), img.getHeight());

	gui.setup();
	
	btnRGBSplit1.addListener(this, &ofApp::setPostProcessMethod<splitRGB1>);
	btnRGBSplit2.addListener(this, &ofApp::setPostProcessMethod<splitRGB2>);
	btnSand.addListener(this, &ofApp::setPostProcessMethod<sand>);
	btnScanLine.addListener(this, &ofApp::setPostProcessMethod<scanLine>);
	btnBlock1.addListener(this, &ofApp::setPostProcessMethod<block1>);
	btnBlock2.addListener(this, &ofApp::setPostProcessMethod<block2>);
	btnDigitalSprite.addListener(this, &ofApp::setPostProcessMethod<digitalStripe>);
	btnIntDigitalSprite.addListener(this, &ofApp::setPostProcessMethod<intDigitalStripe>);

	gui.add(alphaCam.setup("Alpha", 0.3, 0, 1));
	gui.add(btnRGBSplit1.setup("RGB Split V1"));
	gui.add(btnRGBSplit2.setup("RGB Split V2"));
	gui.add(btnScanLine.setup("Scan Line"));
	gui.add(btnSand.setup("Sand"));
	gui.add(btnBlock1.setup("Block V1"));
	gui.add(btnBlock2.setup("Block V2"));
	gui.add(btnDigitalSprite.setup("Digital Stripe"));
	gui.add(btnIntDigitalSprite.setup("Intermidiate Stripe"));

	_Postprocess = splitRGB1;
}

//--------------------------------------------------------------
void ofApp::update(){
	
	videoGrabber.update();

	if (videoGrabber.isFrameNew()) {

		ofImage imgCam = videoGrabber.getPixels();
		Mat matCam = toCv(imgCam);
		//Intermidiate result of merging real-world and provided image, will apply postprocess later
		Mat matMerge(matImg.rows, matImg.cols, CV_8UC3);

		//To roughly record per frame difference, atomic is for thread safe.
		std::atomic<uint64> differentCount = 0;
		cvtColor(matCam, matMask, cv::COLOR_RGB2GRAY);
		threshold(matMask, matMask, 0, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		if (matMaskPre.empty()) {
			matMask.copyTo(matMaskPre);
		}

		//What below does is merging two images and record difference between previous frame.
		//Just use bare threads here.
		std::vector<std::thread> threads(numThreads - 1);
		for (uint64 i = 0; i < numThreads - 1; i++) {

			auto f = [=, &differentCount] {
				for (uint64 j = i * countPerThread; j < (i + 1) * countPerThread; j++) {
					matMerge.data[j * 3] = matImg.data[j * 3] * (1 - alphaCam) + matCam.data[j * 3] * alphaCam;
					matMerge.data[j * 3 + 1] = matImg.data[j * 3 + 1] * (1 - alphaCam) + matCam.data[j * 3 + 1] * alphaCam;
					matMerge.data[j * 3 + 2] = matImg.data[j * 3 + 2] * (1 - alphaCam) + matCam.data[j * 3 + 2] * alphaCam;
					if (matMask.data[j] != matMaskPre.data[j]) {
						differentCount++;
						matMaskPre.data[j] = matMask.data[j];
					}
				}
			};
			threads[i] = std::thread(f);
		}
		for (uint64 j = (numThreads - 1) * countPerThread; j < numThreads * countPerThread; j++) {
			matMerge.data[j * 3] = matImg.data[j*3] * (1 - alphaCam) + matCam.data[j * 3] * alphaCam;
			matMerge.data[j * 3 + 1] = matImg.data[j*3+1] * (1 - alphaCam) + matCam.data[j * 3 + 1] * alphaCam;
			matMerge.data[j * 3 + 2] = matImg.data[j*3+2] * (1 - alphaCam) + matCam.data[j * 3 + 2] * alphaCam;
			if (matMask.data[j] != matMaskPre.data[j]) {
				differentCount++;
				matMaskPre.data[j] = matMask.data[j];
			}
		}
		for (auto& entry : threads) {
			entry.join();
		}
		
		//Process to get final result.
		float intensity =  float(differentCount)/float(matImg.cols * matImg.rows);
		_Postprocess(matResult, matMerge, intensity);
		
	}
}

//--------------------------------------------------------------
void ofApp::draw(){
	drawMat(matResult, 0, 0);
	gui.draw();
}

//Seperate rgb channel horizontally.
void splitRGB1(Mat& OutResult, const Mat& InMat, float intensity) {

	std::vector<std::thread> threads(numThreads - 1);
	int splitAmount = 250 * intensity * ofRandomuf();
	for (int i = 0; i < numThreads - 1; i++) {
		auto f = [=] {
			for (int j =  i * countPerThread; j < (i + 1) * countPerThread; j++) {
				
				OutResult.data[3 * j]     = InMat.data[(((j % InMat.cols + splitAmount) % InMat.cols + (j / InMat.cols) * InMat.cols) * 3)];
				OutResult.data[3 * j + 1] = InMat.data[j * 3 + 1];
				OutResult.data[3 * j + 2] = InMat.data[(((j % InMat.cols - splitAmount + InMat.cols) % InMat.cols + (j / InMat.cols) * InMat.cols) * 3) + 2];
			}
		};
		threads[i] = std::thread(f);
	}
	for (int j = (numThreads - 1) * countPerThread; j < InMat.cols * InMat.rows; j++) {
		
		OutResult.data[3 * j] = InMat.data[(((j % InMat.cols + splitAmount) % InMat.cols + (j / InMat.cols) * InMat.cols) * 3)];
		OutResult.data[3 * j + 1] = InMat.data[j * 3 + 1];
		OutResult.data[3 * j + 2] = InMat.data[(((j % InMat.cols - splitAmount + InMat.cols) % InMat.cols + (j / InMat.cols) * InMat.cols) * 3) + 2];
	}
	for (auto& entry : threads) {
		entry.join();
	}
}

//Seperate rgb channel both horithontally and vertically
void splitRGB2(Mat& OutResult, const Mat& InMat, float intensity) {

	std::vector<std::thread> threads(numThreads - 1);
	int splitAmount = 250 * intensity * ofRandomuf();
	for (int i = 0; i < numThreads - 1; i++) {
		auto f = [=] {
			for (int j = i * countPerThread; j < (i + 1) * countPerThread; j++) {

				OutResult.data[3 * j] = InMat.data[(((j % InMat.cols + splitAmount) % InMat.cols + (j / InMat.cols + splitAmount) % InMat.rows * InMat.cols) * 3)];
				OutResult.data[3 * j + 1] = InMat.data[j * 3 + 1];
				OutResult.data[3 * j + 2] = InMat.data[(((j % InMat.cols - splitAmount + InMat.cols) % InMat.cols + (j / InMat.cols + splitAmount) % InMat.rows * InMat.cols) * 3) + 2];
			}
		};
		threads[i] = std::thread(f);
	}
	for (int j = (numThreads - 1) * countPerThread; j < InMat.cols * InMat.rows; j++) {

		OutResult.data[3 * j] = InMat.data[(((j % InMat.cols + splitAmount) % InMat.cols + (j / InMat.cols + splitAmount) % InMat.rows * InMat.cols) * 3)];
		OutResult.data[3 * j + 1] = InMat.data[j * 3 + 1];
		OutResult.data[3 * j + 2] = InMat.data[(((j % InMat.cols - splitAmount + InMat.cols) % InMat.cols + (j / InMat.cols + splitAmount) % InMat.rows * InMat.cols) * 3) + 2];
	}
	for (auto& entry : threads) {
		entry.join();
	}
}

//Each row takes a random offset to generate such an effect.
void scanLine(Mat& OutResult, const Mat& InMat, float intensity) {

	std::vector<std::thread> threads(numThreads - 1);
	for (int i = 0; i < numThreads - 1; i++) {
		auto f = [=] {
			float jitter = ofRandomuf();
			int row = 0;
			for (int j = i * countPerThread; j < (i + 1) * countPerThread; j++) {
				if (j / InMat.cols != row) {
					jitter = ofRandomuf();
					row = j / InMat.cols;
				}
				int splitAmount = 250 * intensity *  (jitter * 2 - 1);
				OutResult.data[3 * j]	  = InMat.data[(((j % InMat.cols + splitAmount + InMat.cols) % InMat.cols + (j / InMat.cols) * InMat.cols) * 3)];
				OutResult.data[3 * j + 1] = InMat.data[(((j % InMat.cols + splitAmount + InMat.cols) % InMat.cols + (j / InMat.cols) * InMat.cols) * 3) + 1];
				OutResult.data[3 * j + 2] = InMat.data[(((j % InMat.cols + splitAmount + InMat.cols) % InMat.cols + (j / InMat.cols) * InMat.cols) * 3) + 2];
			}
		};
		threads[i] = std::thread(f);
	}

	float jitter = ofRandomuf();
	int row = 0;
	for (int j = (numThreads - 1) * countPerThread; j < InMat.cols * InMat.rows; j++) {
		
		if (j / InMat.cols != row) {
			jitter = ofRandomuf();
			row = j / InMat.cols;
		}
		int splitAmount = 250 * intensity * (jitter * 2 - 1);
		OutResult.data[3 * j]	  = InMat.data[(((j % InMat.cols + splitAmount + InMat.cols) % InMat.cols + (j / InMat.cols) * InMat.cols) * 3)];
		OutResult.data[3 * j + 1] = InMat.data[(((j % InMat.cols + splitAmount + InMat.cols) % InMat.cols + (j / InMat.cols) * InMat.cols) * 3) + 1];
		OutResult.data[3 * j + 2] = InMat.data[(((j % InMat.cols + splitAmount + InMat.cols) % InMat.cols + (j / InMat.cols) * InMat.cols) * 3) + 2];
	}

	for (auto& entry : threads) {
		entry.join();
	}
}

//Each pixel takes a random two dimensional offset.
void sand(Mat& OutResult, const Mat& InMat, float intensity) {
	
	std::vector<std::thread> threads(numThreads - 1);
	for (int i = 0; i < numThreads - 1; i++) {
		auto f = [=] {
			for (int j = i * countPerThread; j < (i + 1) * countPerThread; j++) {

				int splitAmountX = 250 * intensity * (ofRandomuf() * 2 - 1);
				int splitAmountY = 250 * intensity * (ofRandomuf() * 2 - 1);
				OutResult.data[3 * j] = InMat.data[(((j % InMat.cols + splitAmountX + InMat.cols) % InMat.cols     + (j / InMat.cols + splitAmountY + InMat.rows) % InMat.rows * InMat.cols) * 3)];
				OutResult.data[3 * j + 1] = InMat.data[(((j % InMat.cols + splitAmountX + InMat.cols) % InMat.cols + (j / InMat.cols + splitAmountY + InMat.rows) % InMat.rows * InMat.cols) * 3) + 1];
				OutResult.data[3 * j + 2] = InMat.data[(((j % InMat.cols + splitAmountX + InMat.cols) % InMat.cols + (j / InMat.cols + splitAmountY + InMat.rows) % InMat.rows * InMat.cols) * 3) + 2];
			}
		};
		threads[i] = std::thread(f);
	}
	for (int j = (numThreads - 1) * countPerThread; j < InMat.cols * InMat.rows; j++) {
		int splitAmountX = 250 * intensity * (ofRandomuf() * 2 - 1);
		int splitAmountY = 250 * intensity * (ofRandomuf() * 2 - 1);
		OutResult.data[3 * j] = InMat.data[(((j % InMat.cols + splitAmountX + InMat.cols) % InMat.cols	   + (j / InMat.cols + splitAmountY + InMat.rows) % InMat.rows * InMat.cols) * 3)];
		OutResult.data[3 * j + 1] = InMat.data[(((j % InMat.cols + splitAmountX + InMat.cols) % InMat.cols + (j / InMat.cols + splitAmountY + InMat.rows) % InMat.rows * InMat.cols) * 3) + 1];
		OutResult.data[3 * j + 2] = InMat.data[(((j % InMat.cols + splitAmountX + InMat.cols) % InMat.cols + (j / InMat.cols + splitAmountY + InMat.rows) % InMat.rows * InMat.cols) * 3) + 2];
	}

	for (auto& entry : threads) {
		entry.join();
	}
}

//Each block takes a random rgb split value.
void block1(Mat& OutResult, const Mat& InMat, float intensity) {
	
	int blockCount = 10;
	int blockWidth = InMat.cols / blockCount;
	int blockHeight = InMat.rows / blockCount;

	Mat randomNoise(blockCount, blockCount, CV_8UC1);

	for (int i = 0; i < blockCount * blockCount; i++) {
		randomNoise.data[i] = ofRandomuf() * 255;
	}

	std::vector<std::thread> threads(numThreads - 1);
	for (int i = 0; i < numThreads - 1; i++) {
		auto f = [=] {
			for (int j = i * countPerThread; j < (i + 1) * countPerThread; j++) {
				int rowIndex = j / InMat.cols;
				int columnIndex = j % InMat.cols;
				float random = randomNoise.data[(rowIndex/blockHeight) * randomNoise.cols + columnIndex / blockWidth];
				int splitAmount = 5 * intensity * random;
				OutResult.data[3 * j] = InMat.data[((columnIndex + splitAmount + InMat.cols)%InMat.cols + rowIndex * InMat.cols) * 3];
				OutResult.data[3 * j + 1] = InMat.data[j * 3 + 1];
				OutResult.data[3 * j + 2] = InMat.data[((columnIndex - splitAmount + InMat.cols) % InMat.cols + rowIndex * InMat.cols) * 3 + 2];
			}
		};
		threads[i] = std::thread(f);
	}
	for (int j =  (numThreads - 1) * countPerThread; j < numThreads * countPerThread; j++) {
		int rowIndex = j / InMat.cols;
		int columnIndex = j % InMat.cols;
		float random = randomNoise.data[(rowIndex / blockHeight) * randomNoise.cols + columnIndex / blockWidth];
		int splitAmount = 5 * intensity * random;
		OutResult.data[3 * j] = InMat.data[((columnIndex + splitAmount + InMat.cols) % InMat.cols + rowIndex * InMat.cols) * 3];
		OutResult.data[3 * j + 1] = InMat.data[j * 3 + 1];
		OutResult.data[3 * j + 2] = InMat.data[((columnIndex - splitAmount + InMat.cols) % InMat.cols + rowIndex * InMat.cols) * 3 + 2];
	}

	for (auto& entry : threads) {
		entry.join();
	}
}

//Add a threhold to previous block, such that can generate random blocks.
void block2(Mat& OutResult, const Mat& InMat, float intensity) {

	int blockCount = 10;
	int blockWidth = InMat.cols / blockCount;
	int blockHeight = InMat.rows / blockCount;

	Mat randomNoise(blockCount, blockCount, CV_8UC1);

	for (int i = 0; i < blockCount * blockCount; i++) {
		randomNoise.data[i] = ofRandomuf() * 255;
		if (ofRandomuf() > intensity) {
			randomNoise.data[i] = 0;
		}
	}

	std::vector<std::thread> threads(numThreads - 1);
	for (int i = 0; i < numThreads - 1; i++) {
		auto f = [=] {
			for (int j = i * countPerThread; j < (i + 1) * countPerThread; j++) {
				int rowIndex = j / InMat.cols;
				int columnIndex = j % InMat.cols;
				float random = randomNoise.data[(rowIndex / blockHeight) * randomNoise.cols + columnIndex / blockWidth];
				int splitAmount = 5 * intensity * random;
				OutResult.data[3 * j] = InMat.data[((columnIndex + splitAmount + InMat.cols) % InMat.cols + rowIndex * InMat.cols) * 3];
				OutResult.data[3 * j + 1] = InMat.data[j * 3 + 1];
				OutResult.data[3 * j + 2] = InMat.data[((columnIndex - splitAmount + InMat.cols) % InMat.cols + rowIndex * InMat.cols) * 3 + 2];
			}
		};
		threads[i] = std::thread(f);
	}
	for (int j = (numThreads - 1) * countPerThread; j < numThreads * countPerThread; j++) {
		int rowIndex = j / InMat.cols;
		int columnIndex = j % InMat.cols;
		float random = randomNoise.data[(rowIndex / blockHeight) * randomNoise.cols + columnIndex / blockWidth];
		int splitAmount = 5 * intensity * random;
		OutResult.data[3 * j] = InMat.data[((columnIndex + splitAmount + InMat.cols) % InMat.cols + rowIndex * InMat.cols) * 3];
		OutResult.data[3 * j + 1] = InMat.data[j * 3 + 1];
		OutResult.data[3 * j + 2] = InMat.data[((columnIndex - splitAmount + InMat.cols) % InMat.cols + rowIndex * InMat.cols) * 3 + 2];
	}

	for (auto& entry : threads) {
		entry.join();
	}
}

//Generate a random color noise image first, then do random selection.
void intDigitalStripe(Mat& OutResult, const Mat& InMat, float intensity) {

	//generate noise image
	int rowClusterNum = InMat.rows / 30;
	Mat noiseMat(InMat.rows / rowClusterNum + 1, InMat.cols, CV_8UC4);

	int glitchRow = 0;
	int countWidth = 0;
	int randRed;
	int randGreen;
	int randBlue;
	int alpha;

	for (int i = 0; i < noiseMat.rows; i++) {

		for (int j = 0; j < noiseMat.cols; j++) {

			if (countWidth >= glitchRow) {
				countWidth = 0;
				glitchRow = ofRandomuf() * InMat.cols * (1 - intensity);
				randRed = ofRandomuf() * 255;
				randGreen = ofRandomuf() * 255;
				randBlue = ofRandomuf() * 255;
				alpha = ofRandomuf() > intensity ? 0 : 255;
			}
			noiseMat.data[(j + i * noiseMat.cols) * 4] = randRed;
			noiseMat.data[(j + i * noiseMat.cols) * 4 + 1] = randGreen;
			noiseMat.data[(j + i * noiseMat.cols) * 4 + 2] = randBlue;
			noiseMat.data[(j + i * noiseMat.cols) * 4 + 3] = alpha;
			countWidth++;
		}
	}

	std::vector<std::thread> threads(numThreads - 1);
	for (int i = 0; i < numThreads - 1; i++) {

		auto f = [=] {
			for (int j = i * countPerThread; j < (i + 1) * countPerThread; j++) {
				int alpha = noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4 + 3];
				OutResult.data[3 * j] = alpha * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4];
				OutResult.data[3 * j + 1] = alpha * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4 + 1];
				OutResult.data[3 * j + 2] = alpha * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4 + 2];
			}
		};
		threads[i] = std::thread(f);
	}

	for (int j = (numThreads - 1) * countPerThread; j < InMat.cols * InMat.rows; j++) {
		int alpha = noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4 + 3];
		OutResult.data[3 * j] = alpha * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 3];
		OutResult.data[3 * j + 1] = alpha * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 3 + 1];
		OutResult.data[3 * j + 2] = alpha * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 3 + 2];
	}

	for (auto& entry : threads) {
		entry.join();
	}
}

//Apply the previous generated image to the merged result.
//This case I do merge and inverse. There could be various visual effect.
void digitalStripe(Mat& OutResult, const Mat& InMat, float intensity) {

	//generate noise texture
	int rowClusterNum = InMat.rows / 25;
	Mat noiseMat(InMat.rows / rowClusterNum + 1, InMat.cols, CV_8UC4);
	
	int glitchRow = 0;
	int countWidth = 0;
	int randRed;
	int randGreen;
	int randBlue;
	int alpha;

	for (int i = 0; i < noiseMat.rows; i++) {
		
		for (int j = 0; j < noiseMat.cols; j++) {
			
			if (countWidth >= glitchRow) {
				countWidth = 0;
				glitchRow = ofRandomuf() * InMat.cols * (1 - intensity);
				randRed	  = ofRandomuf() * 255;
				randGreen = ofRandomuf() * 255;
				randBlue  = ofRandomuf() * 255;
				alpha =  ofRandomuf() > intensity ? 0 : 255;
			}
			noiseMat.data[(j + i * noiseMat.cols) * 4] = randRed;
			noiseMat.data[(j + i * noiseMat.cols) * 4 + 1] = randGreen;
			noiseMat.data[(j + i * noiseMat.cols) * 4 + 2] = randBlue;
			noiseMat.data[(j + i * noiseMat.cols) * 4 + 3] = alpha;
			countWidth++;
		}
	}

	std::vector<std::thread> threads(numThreads - 1);

	for (int i = 0; i < numThreads - 1; i++) {

		auto f = [=] {
			for (int j = i * countPerThread; j < (i + 1) * countPerThread; j++) {
				if (noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4 + 3]) {
					OutResult.data[3 * j] = 255 - (InMat.data[3 * j] * 0.5 + 0.5 * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4]);
					OutResult.data[3 * j + 1] = 255 - (InMat.data[3 * j + 1] * 0.5 + 0.5 * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4 + 1]);
					OutResult.data[3 * j + 2] = 255 - (InMat.data[3 * j + 2] * 0.5 + 0.5 * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4 + 2]);
				}
				else {
					OutResult.data[3 * j] = InMat.data[3 * j];
					OutResult.data[3 * j + 1] = InMat.data[3 * j + 1];
					OutResult.data[3 * j + 2] = InMat.data[3 * j + 2];
				}
			}
		};
		threads[i] = std::thread(f);
	}

	for (int j = (numThreads - 1) * countPerThread; j < InMat.cols * InMat.rows; j++) {
		if (noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4 + 3]) {
			OutResult.data[3 * j] = 255 - (InMat.data[3 * j] * 0.5 + 0.5 * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4]);
			OutResult.data[3 * j + 1] = 255 - (InMat.data[3 * j + 1] * 0.5 + 0.5 * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4 + 1]);
			OutResult.data[3 * j + 2] = 255 - (InMat.data[3 * j + 2] * 0.5 + 0.5 * noiseMat.data[(InMat.cols * ((j / InMat.cols) / rowClusterNum) + j % InMat.cols) * 4 + 2]);
		}
		else {
			OutResult.data[3 * j] = InMat.data[3 * j];
			OutResult.data[3 * j + 1] = InMat.data[3 * j + 1];
			OutResult.data[3 * j + 2] = InMat.data[3 * j + 2];
		}
	}

	for (auto& entry : threads) {
		entry.join();
	}
}

