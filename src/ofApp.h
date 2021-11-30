#pragma once

#include "ofMain.h"
#include "ofxCv.h"
#include "ofxOpenCv.h"
#include "ofxGui.h"

using namespace ofxCv;
using namespace cv;

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
		
		//Input image
		ofImage img;
		Mat matImg;
		Mat matResult;
		Mat matMask;
		Mat matMaskPre;

		//Video input.
		ofVideoGrabber videoGrabber;
		ofxPanel gui;
		ofxFloatSlider alphaCam;

		//Buttons
		ofxButton btnRGBSplit1;
		ofxButton btnRGBSplit2;
		ofxButton btnSand;
		ofxButton btnScanLine;
		ofxButton btnBlock1;
		ofxButton btnBlock2;
		ofxButton btnLine;
		ofxButton btnDigitalSprite;
		ofxButton btnIntDigitalSprite;

		//Postprocess Func
		void (*_Postprocess)(Mat&, const Mat&, float);
		template<void (*Func)(Mat&, const Mat&, float)>
			void setPostProcessMethod() { _Postprocess = Func; };
		
};
