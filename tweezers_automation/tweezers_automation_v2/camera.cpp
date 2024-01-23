#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <bgapi2_genicam.hpp>
#include <iostream>
#include "camera.h"
#include "spot_manager.h"
#include <chrono>
//#include "spot_manager.h"

#define CAM_FRAME_RATE 250
#define BEAD_DETECT_RATE 100
#define BEAD_TRACKING_RATE 100

extern std::mutex m;
extern std::mutex k;
extern std::mutex g;
extern std::vector<cv::KeyPoint> keypoints;
extern cv::Mat cam_img;

// Identifies and uniquely labels all detected beads between consecutive frames
// Used for PID control of bead position
void bead_tracking(SpotManager* spotManager) {
    while (true) {
        int num_tracked = 0;

        {
            std::lock_guard<std::mutex> lock_g(g);
            std::lock_guard<std::mutex> lock_k(k);

            for (auto& bead : keypoints) {
                int x_detect = bead.pt.x;
                int y_detect = bead.pt.y;
                int radius = 15; // pixels
                int y_min = std::max(0, y_detect - radius);
                int y_max = std::min(480, y_detect + radius);
                int x_min = std::max(0, x_detect - radius);
                int x_max = std::min(640, x_detect + radius);

                for (int i = x_min; i < x_max; i++) {
                    for (int j = y_min; j < y_max; j++) {
                        if (spotManager->grid[j][i].assigned) {
                            //std::cout << "Assigned bead at: (" << i << ", " << j << ") ";
                            //std::cout << "Detected bead at: (" << x_detect << ", " << y_detect << ")" << std::endl;
                            num_tracked += 1;
                            
                            double control_x = spotManager->grid[j][i].pid_x->calculate(i, x_detect);
                            double control_y = spotManager->grid[j][i].pid_y->calculate(j, y_detect);
                            //std::cout << "PID output: (" << control_x + i << ", " << control_y + j << ")" << std::endl;

                            spotManager->grid[j][i].set_new_pos(control_y + j, control_x + i);  // (slm_x, slm_y)
                            spotManager->update_traps();

                        }
                    }
                }
            }
        }
        //std::cout << keypoints.size() << " ";
        //std::cout << num_tracked << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(BEAD_TRACKING_RATE));
    }
}

void detect_beads() {

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(BEAD_DETECT_RATE));
        auto startTime = std::chrono::high_resolution_clock::now();
        cv::Mat oriimg;

        {
            std::lock_guard<std::mutex> lock_m(m);
            oriimg = cam_img.clone();
            cv::cvtColor(cam_img.clone(), oriimg, cv::COLOR_BGR2GRAY);
        } // lock_m is automatically released when it goes out of scope

        if (oriimg.empty()) {
            std::cout << "Could not open or find the image" << std::endl;
        }

        cv::Mat img;
        cv::Mat img1;
        cv::Mat img2;
        cv::Mat edges;
        int h = oriimg.cols; int w = oriimg.rows;

        //cv::subtract(oriimg, b, img1, cv::noArray(), CV_64FC1);
        //cv::divide(img1, a, img2);

        //Step 1: detecting beads

        oriimg.convertTo(img, CV_8UC1);
        cv::Mat Step1;

        int offset = 2;

        cv::equalizeHist(img, Step1);

        cv::medianBlur(Step1, Step1, 3);
        cv::copyMakeBorder(Step1, Step1, offset, offset, offset, offset, cv::BORDER_CONSTANT, cv::Scalar(255));

        cv::SimpleBlobDetector::Params params;
        params.minArea = 300;
        params.minCircularity = 0.85;
        params.filterByArea = 1;
        params.filterByInertia = 1;
        params.filterByCircularity = 1;
        params.filterByColor = 1;

        {
            std::lock_guard<std::mutex> lock_k(k);
            cv::Ptr<cv::SimpleBlobDetector> blobDetector = cv::SimpleBlobDetector::create(params);
            blobDetector->detect(Step1, keypoints);
        } // lock_k is automatically released when it goes out of scope

        cv::Mat imgWithKeypoints;

        for (int i = 0; i < keypoints.size(); i++) {
            keypoints[i].pt = cv::Point(keypoints[i].pt.x, keypoints[i].pt.y);
            //std::cout << keypoints[i].pt;  // print out coordinates of beads
        }

        //cv::imshow("Camera", cam_img);
        //std::cout << "Detected " << keypoints.size() << " beads" << std::endl;

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        //std::cout << "Execution Time: " << duration.count() << " seconds" << std::endl;
        //std::cout << std::endl;
    }
    
    
}

int get_img_offline_test(SpotManager* spotManager) {
    std::string videoFilePath = "C:\\Users\\Tommy\\Desktop\\Tweezers Videos\\testing.mp4";

    // Open the video file
    cv::VideoCapture videoCapture(videoFilePath);

    // Check if the video file is opened successfully
    if (!videoCapture.isOpened()) {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return -1;
    }

    // Create a window for displaying the video frames
    cv::namedWindow("Video Player", cv::WINDOW_NORMAL);

    // Main loop to read and display frames
    while (true) {

        std::vector<cv::KeyPoint> trap_locations;
        // Iterate through the map and convert elements to KeyPoint
        for (const auto& entry : spotManager->trapped_beads) {
            // Extract x, y coordinates from the pair
            int x = entry.first.second;
            int y = entry.first.first;
            Spot spot = spotManager->grid[y][x];

            x = -spot.vals[1] / 0.1875;
            y = spot.vals[0] / 0.1875;

            // Create a cv::KeyPoint object
            cv::KeyPoint keypoint(x, y, 20); // You might want to adjust the size parameter (third argument)

            // Add the KeyPoint to the vector
            trap_locations.push_back(keypoint);
        }


        // Read a frame from the video file
        cv::Mat frame;
        videoCapture >> frame;

        // Check if the frame is empty (end of video)
        if (frame.empty()) {
            std::cout << "End of video." << std::endl;
            break;
        }
        cv::Mat imgWithKeypoints;
        cv::Mat imgWithTrapLocations;
        {
            std::unique_lock<std::mutex> lock_m(m);
            cam_img = frame.clone(); 
            cv::drawKeypoints(cam_img, keypoints, imgWithKeypoints, cv::Scalar(255, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        }
       
        cv::drawKeypoints(imgWithKeypoints, trap_locations, imgWithTrapLocations, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        cv::imshow("Video Player", imgWithTrapLocations);

        cv::waitKey(50);

    }

    // Release the video capture object and close the window
    videoCapture.release();
    cv::destroyAllWindows();

    return 0;
}

int get_img(SpotManager* spotManager) {
    // DECLARATIONS OF VARIABLES
    BGAPI2::ImageProcessor* imgProcessor = NULL;

    BGAPI2::SystemList* systemList = NULL;
    BGAPI2::System* pSystem = NULL;
    BGAPI2::String sSystemID;

    BGAPI2::InterfaceList* interfaceList = NULL;
    BGAPI2::Interface* pInterface = NULL;
    BGAPI2::String sInterfaceID;

    BGAPI2::DeviceList* deviceList = NULL;
    BGAPI2::Device* pDevice = NULL;
    BGAPI2::String sDeviceID;

    BGAPI2::DataStreamList* datastreamList = NULL;
    BGAPI2::DataStream* pDataStream = NULL;
    BGAPI2::String sDataStreamID;

    BGAPI2::BufferList* bufferList = NULL;
    BGAPI2::Buffer* pBuffer = NULL;
    BGAPI2::String sBufferID;
    int returncode = 0;

    // OPENCV VARIABLE DECLARATIONS
    cv::VideoWriter cvVideoCreator;                 // Create OpenCV video creator
    cv::String videoFileName = "openCvVideo.avi";   // Define video filename
    cv::Size frameSize = cv::Size(2048, 1088);      // Define video frame size (frame width x height)
    cvVideoCreator.open(videoFileName, cv::VideoWriter::fourcc('D', 'I', 'V', 'X'), 20, frameSize, true); // set the codec type and frame rate

    // Load image processor
    try {
        imgProcessor = new BGAPI2::ImageProcessor();
        std::cout << "ImageProcessor version:    " << imgProcessor->GetVersion() << std::endl;
        if (imgProcessor->GetNodeList()->GetNodePresent("DemosaicingMethod") == true) {
            // possible values: NearestNeighbor, Bilinear3x3, Baumer5x5
            imgProcessor->GetNodeList()->GetNode("DemosaicingMethod")->SetString("NearestNeighbor");
            std::cout << "    Demosaicing method:    "
                << imgProcessor->GetNodeList()->GetNode("DemosaicingMethod")->GetString() << std::endl;
        }
        std::cout << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }


    std::cout << "SYSTEM LIST" << std::endl;
    std::cout << "###########" << std::endl << std::endl;

    // COUNTING AVAILABLE SYSTEMS (TL producers)
    try {
        systemList = BGAPI2::SystemList::GetInstance();
        systemList->Refresh();
        std::cout << "5.1.2   Detected systems:  " << systemList->size() << std::endl;

        // SYSTEM DEVICE INFORMATION
        for (BGAPI2::SystemList::iterator sysIterator = systemList->begin();
            sysIterator != systemList->end();
            sysIterator++) {

            /*std::cout << "  5.2.1   System Name:     " << sysIterator->GetFileName() << std::endl;
            std::cout << "          System Type:     " << sysIterator->GetTLType() << std::endl;
            std::cout << "          System Version:  " << sysIterator->GetVersion() << std::endl;
            std::cout << "          System PathName: " << sysIterator->GetPathName() << std::endl << std::endl;*/
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        /*std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;*/
    }


    // OPEN THE FIRST SYSTEM IN THE LIST WITH A CAMERA CONNECTED
    try {
        for (BGAPI2::SystemList::iterator sysIterator = systemList->begin();
            sysIterator != systemList->end();
            sysIterator++) {
            /*std::cout << "SYSTEM" << std::endl;
            std::cout << "######" << std::endl << std::endl;*/

            try {
                sysIterator->Open();
                std::cout << "5.1.3   Open next system " << std::endl;
                std::cout << "  5.2.1   System Name:     " << sysIterator->GetFileName() << std::endl;
                std::cout << "          System Type:     " << sysIterator->GetTLType() << std::endl;
                std::cout << "          System Version:  " << sysIterator->GetVersion() << std::endl;
                std::cout << "          System PathName: " << sysIterator->GetPathName() << std::endl
                    << std::endl;
                sSystemID = sysIterator->GetID();
                /*std::cout << "        Opened system - NodeList Information " << std::endl;
                std::cout << "          GenTL Version:   "
                    << sysIterator->GetNode("GenTLVersionMajor")->GetValue() << "."
                    << sysIterator->GetNode("GenTLVersionMinor")->GetValue() << std::endl << std::endl;*/

                    /* std::cout << "INTERFACE LIST" << std::endl;
                     std::cout << "##############" << std::endl << std::endl;*/

                try {
                    interfaceList = sysIterator->GetInterfaces();
                    // COUNT AVAILABLE INTERFACES
                    interfaceList->Refresh(100);  // timeout of 100 msec
                    /*std::cout << "5.1.4   Detected interfaces: " << interfaceList->size() << std::endl;*/

                    // INTERFACE INFORMATION
                    for (BGAPI2::InterfaceList::iterator ifIterator = interfaceList->begin();
                        ifIterator != interfaceList->end();
                        ifIterator++) {
                        /*std::cout << "  5.2.2   Interface ID:      "
                            << ifIterator->GetID() << std::endl;
                        std::cout << "          Interface Type:    "
                            << ifIterator->GetTLType() << std::endl;
                        std::cout << "          Interface Name:    "
                            << ifIterator->GetDisplayName() << std::endl << std::endl;*/
                    }
                }
                catch (BGAPI2::Exceptions::IException& ex) {
                    returncode = (returncode == 0) ? 1 : returncode;
                    /*std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
                    std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
                    std::cout << "in function:      " << ex.GetFunctionName() << std::endl;*/
                }


                /*std::cout << "INTERFACE" << std::endl;
                std::cout << "#########" << std::endl << std::endl;*/

                // OPEN THE NEXT INTERFACE IN THE LIST
                try {
                    for (BGAPI2::InterfaceList::iterator ifIterator = interfaceList->begin();
                        ifIterator != interfaceList->end();
                        ifIterator++) {
                        try {
                            /*std::cout << "5.1.5   Open interface " << std::endl;
                            std::cout << "  5.2.2   Interface ID:      "
                                << ifIterator->GetID() << std::endl;
                            std::cout << "          Interface Type:    "
                                << ifIterator->GetTLType() << std::endl;
                            std::cout << "          Interface Name:    "
                                << ifIterator->GetDisplayName() << std::endl;*/
                            ifIterator->Open();
                            // search for any camera is connetced to this interface
                            deviceList = ifIterator->GetDevices();
                            deviceList->Refresh(100);
                            if (deviceList->size() == 0) {
                                /* std::cout << "5.1.13   Close interface (" << deviceList->size() << " cameras found) "
                                     << std::endl << std::endl;*/
                                ifIterator->Close();
                            }
                            else {
                                sInterfaceID = ifIterator->GetID();
                                /*std::cout << "   " << std::endl;
                                std::cout << "        Opened interface - NodeList Information" << std::endl;*/
                                if (ifIterator->GetTLType() == "GEV") {
                                    /* std::cout << "          GevInterfaceSubnetIPAddress: "
                                         << ifIterator->GetNode("GevInterfaceSubnetIPAddress")->GetValue()
                                         << std::endl;
                                     std::cout << "          GevInterfaceSubnetMask:      "
                                         << ifIterator->GetNode("GevInterfaceSubnetMask")->GetValue()
                                         << std::endl;*/
                                }
                                if (ifIterator->GetTLType() == "U3V") {
                                    // std::cout << "          NodeListCount:     "
                                    // << ifIterator->GetNodeList()->GetNodeCount() << std::endl;
                                }
                                std::cout << "  " << std::endl;
                                break;
                            }
                        }
                        catch (BGAPI2::Exceptions::ResourceInUseException& ex) {
                            returncode = (returncode == 0) ? 1 : returncode;
                            /*std::cout << " Interface " << ifIterator->GetID() << " already opened " << std::endl;
                            std::cout << " ResourceInUseException: " << ex.GetErrorDescription() << std::endl;*/
                        }
                    }
                }
                catch (BGAPI2::Exceptions::IException& ex) {
                    returncode = (returncode == 0) ? 1 : returncode;
                    /*std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
                    std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
                    std::cout << "in function:      " << ex.GetFunctionName() << std::endl;*/
                }


                // if a camera is connected to the system interface then leave the system loop
                if (sInterfaceID != "") {
                    break;
                }
            }
            catch (BGAPI2::Exceptions::ResourceInUseException& ex) {
                returncode = (returncode == 0) ? 1 : returncode;
                /*std::cout << " System " << sysIterator->GetID() << " already opened " << std::endl;
                std::cout << " ResourceInUseException: " << ex.GetErrorDescription() << std::endl;*/
            }
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        /* std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
         std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
         std::cout << "in function:      " << ex.GetFunctionName() << std::endl;*/
    }

    if (sSystemID == "") {
        /* std::cout << " No System found " << std::endl;
         std::cout << std::endl << "End" << std::endl << "Input any number to close the program:";*/
        int endKey = 0;
        std::cin >> endKey;
        BGAPI2::SystemList::ReleaseInstance();
        // RELEASE IMAGE PROCESSOR
        delete imgProcessor;
        return returncode;
    }
    else {
        pSystem = (*systemList)[sSystemID];
    }


    if (sInterfaceID == "") {
        /*std::cout << " No camera found " << sInterfaceID << std::endl;
        std::cout << std::endl << "End" << std::endl << "Input any number to close the program:";*/
        int endKey = 0;
        std::cin >> endKey;
        pSystem->Close();
        BGAPI2::SystemList::ReleaseInstance();
        // RELEASE IMAGE PROCESSOR
        delete imgProcessor;
        return returncode;
    }
    else {
        pInterface = (*interfaceList)[sInterfaceID];
    }


    /* std::cout << "DEVICE LIST" << std::endl;
     std::cout << "###########" << std::endl << std::endl;*/

    try {
        // COUNTING AVAILABLE CAMERAS
        deviceList = pInterface->GetDevices();
        deviceList->Refresh(100);
        std::cout << "5.1.6   Detected devices:         " << deviceList->size() << std::endl;

        // DEVICE INFORMATION BEFORE OPENING
        for (BGAPI2::DeviceList::iterator devIterator = deviceList->begin();
            devIterator != deviceList->end();
            devIterator++) {
            /*std::cout << "  5.2.3   Device DeviceID:        "
                << devIterator->GetID() << std::endl;
            std::cout << "          Device Model:           "
                << devIterator->GetModel() << std::endl;
            std::cout << "          Device SerialNumber:    "
                << devIterator->GetSerialNumber() << std::endl;
            std::cout << "          Device Vendor:          "
                << devIterator->GetVendor() << std::endl;
            std::cout << "          Device TLType:          "
                << devIterator->GetTLType() << std::endl;
            std::cout << "          Device AccessStatus:    "
                << devIterator->GetAccessStatus() << std::endl;
            std::cout << "          Device UserID:          "
                << devIterator->GetDisplayName() << std::endl << std::endl;*/
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        /*std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;*/
    }


    /*std::cout << "DEVICE" << std::endl;
    std::cout << "######" << std::endl << std::endl;*/

    // OPEN THE FIRST CAMERA IN THE LIST
    try {
        for (BGAPI2::DeviceList::iterator devIterator = deviceList->begin();
            devIterator != deviceList->end();
            devIterator++) {
            try {
                devIterator->Open();
                sDeviceID = devIterator->GetID();
                break;
            }
            catch (BGAPI2::Exceptions::ResourceInUseException& ex) {
                returncode = (returncode == 0) ? 1 : returncode;
                std::cout << " Device  " << devIterator->GetID() << " already opened " << std::endl;
                std::cout << " ResourceInUseException: " << ex.GetErrorDescription() << std::endl;
            }
            catch (BGAPI2::Exceptions::AccessDeniedException& ex) {
                returncode = (returncode == 0) ? 1 : returncode;
                std::cout << " Device  " << devIterator->GetID() << " already opened " << std::endl;
                std::cout << " AccessDeniedException " << ex.GetErrorDescription() << std::endl;
            }
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    if (sDeviceID == "") {
        std::cout << " No Device found " << sDeviceID << std::endl;
        std::cout << std::endl << "End" << std::endl << "Input any number to close the program:";
        int endKey = 0;
        std::cin >> endKey;
        pInterface->Close();
        pSystem->Close();
        BGAPI2::SystemList::ReleaseInstance();
        // RELEASE IMAGE PROCESSOR
        delete imgProcessor;
        return returncode;
    }
    else {
        pDevice = (*deviceList)[sDeviceID];
    }


    std::cout << "DEVICE PARAMETER SETUP" << std::endl;
    std::cout << "######################" << std::endl << std::endl;

    try {
        // SET TRIGGER MODE OFF (FreeRun)
        pDevice->GetRemoteNode("TriggerMode")->SetString("Off");
        std::cout << "         TriggerMode:             "
            << pDevice->GetRemoteNode("TriggerMode")->GetValue() << std::endl;
        std::cout << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }


    std::cout << "DATA STREAM LIST" << std::endl;
    std::cout << "################" << std::endl << std::endl;

    try {
        // COUNTING AVAILABLE DATASTREAMS
        datastreamList = pDevice->GetDataStreams();
        datastreamList->Refresh();
        std::cout << "5.1.8   Detected datastreams:     " << datastreamList->size() << std::endl;

        // DATASTREAM INFORMATION BEFORE OPENING
        for (BGAPI2::DataStreamList::iterator dstIterator = datastreamList->begin();
            dstIterator != datastreamList->end();
            dstIterator++) {
            std::cout << "  5.2.4   DataStream ID:          " << dstIterator->GetID() << std::endl << std::endl;
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }


    std::cout << "DATA STREAM" << std::endl;
    std::cout << "###########" << std::endl << std::endl;

    // OPEN THE FIRST DATASTREAM IN THE LIST
    try {
        if (datastreamList->size() > 0) {
            pDataStream = (*datastreamList)[0];
            std::cout << "5.1.9   Open first datastream " << std::endl;
            std::cout << "          DataStream ID:          " << pDataStream->GetID() << std::endl << std::endl;
            pDataStream->Open();
            sDataStreamID = pDataStream->GetID();
            std::cout << "        Opened datastream - NodeList Information" << std::endl;
            std::cout << "          StreamAnnounceBufferMinimum:  "
                << pDataStream->GetNode("StreamAnnounceBufferMinimum")->GetValue() << std::endl;
            if (pDataStream->GetTLType() == "GEV") {
                std::cout << "          StreamDriverModel:            "
                    << pDataStream->GetNode("StreamDriverModel")->GetValue() << std::endl;
            }
            std::cout << "  " << std::endl;
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    if (sDataStreamID == "") {
        std::cout << " No DataStream found" << std::endl;
        std::cout << std::endl << "End" << std::endl << "Input any number to close the program:";
        int endKey = 0;
        std::cin >> endKey;
        pDevice->Close();
        pInterface->Close();
        pSystem->Close();
        BGAPI2::SystemList::ReleaseInstance();
        // RELEASE IMAGE PROCESSOR
        delete imgProcessor;
        return returncode;
    }

    std::cout << "BUFFER LIST" << std::endl;
    std::cout << "###########" << std::endl << std::endl;

    try {
        // BufferList
        bufferList = pDataStream->GetBufferList();

        // 4 buffers using internal buffer mode
        for (int i = 0; i < 4; i++) {
            pBuffer = new BGAPI2::Buffer();
            bufferList->Add(pBuffer);
        }
        std::cout << "5.1.10   Announced buffers:       " << bufferList->GetAnnouncedCount() << " using "
            << pBuffer->GetMemSize() * bufferList->GetAnnouncedCount() << " [bytes]" << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    try {
        for (BGAPI2::BufferList::iterator bufIterator = bufferList->begin();
            bufIterator != bufferList->end();
            bufIterator++) {
            bufIterator->QueueBuffer();
        }
        std::cout << "5.1.11   Queued buffers:          " << bufferList->GetQueuedCount() << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }
    std::cout << " " << std::endl;

    std::cout << "CAMERA START" << std::endl;
    std::cout << "############" << std::endl << std::endl;

    // START DataStream acquisition
    try {
        pDataStream->StartAcquisition(4);
        std::cout << "5.1.12   DataStream started " << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    // START CAMERA
    try {
        std::cout << "5.1.12   " << pDevice->GetModel() << " started " << std::endl;
        pDevice->GetRemoteNode("AcquisitionStart")->Execute();
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    // CAPTURE 8 IMAGES
    //std::cout << " " << std::endl;
    //std::cout << "CAPTURE & TRANSFORM 4 IMAGES" << std::endl;
    //std::cout << "############################" << std::endl << std::endl;

    BGAPI2::Buffer* pBufferFilled = NULL;
    try {
        BGAPI2::Image* pImage = imgProcessor->CreateImage();
        BGAPI2::Node* pPixelFormatInfoSelector = imgProcessor->GetNode("PixelFormatInfoSelector");
        BGAPI2::Node* pBytesPerPixel = imgProcessor->GetNode("BytesPerPixel");

        // create OpenCV window ----
        cv::namedWindow("OpenCV window: Cam");

        while (pDataStream->GetIsGrabbing()) {
            pBufferFilled = pDataStream->GetFilledBuffer(1000);  // timeout 1000 msec
            if (pBufferFilled == NULL) {
                std::cout << "Error: Buffer Timeout after 1000 msec" << std::endl << std::endl;
            }
            else if (pBufferFilled->GetIsIncomplete() == true) {
                //std::cout << "Error: Image is incomplete" << std::endl << std::endl;
                // queue buffer again
                pBufferFilled->QueueBuffer();
            }
            else {
                //std::cout << " Image " << std::setw(5) << pBufferFilled->GetFrameID()
                //   << " received in memory address " << std::hex << pBufferFilled->GetMemPtr()
                //   << std::dec << std::endl;

                // create an image object from the filled buffer and convert it
                BGAPI2::Image* pTransformImage = NULL;
                pImage->Init(pBufferFilled);

                BGAPI2::String sPixelFormat = pImage->GetPixelformat();
                /*
                std::cout << "  pImage.Pixelformat:             "
                    << pImage->GetPixelformat() << std::endl;
                std::cout << "  pImage.Width:                   "
                    << pImage->GetWidth() << std::endl;
                std::cout << "  pImage.Height:                  "
                    << pImage->GetHeight() << std::endl;
                std::cout << "  pImage.Buffer:                  "
                    << std::hex << pImage->GetBuffer() << std::dec << std::endl;
                */
                pPixelFormatInfoSelector->SetValue(sPixelFormat);
                double fBytesPerPixel = pBytesPerPixel->GetAvailable() ? pBytesPerPixel->GetDouble() : 0.0;

                //std::cout << "  Bytes per image:                "
                 //   << static_cast<unsigned int>((pImage->GetWidth()) * (pImage->GetHeight()) * fBytesPerPixel)
                  //  << std::endl;
                //std::cout << "  Bytes per pixel:                "
                 //   << fBytesPerPixel << std::endl;

                // display first 6 pixel values of first 6 lines of the image
                // ========================================================================
                unsigned char* imageBuffer = (unsigned char*)pImage->GetBuffer();

                //std::cout << "  Address" << std::endl;
                // set display for uppercase hex numbers filled with '0'
                //std::cout << std::uppercase << std::setfill('0') << std::hex;
                for (int j = 0; j < 6; j++) {  // first 6 lines
                    void* imageBufferAddress = &imageBuffer[static_cast<int>(pImage->GetWidth() * j * fBytesPerPixel)];
                    //std::cout << "  " << std::setw(8) << imageBufferAddress << " ";
                    for (int k = 0; k < static_cast<int>(6 * fBytesPerPixel); k++) {  // bytes of first 6 pixels
                        //std::cout << " " << std::setw(2)
                            //<< static_cast<int>(imageBuffer[static_cast<int>(pImage->GetWidth() * j * fBytesPerPixel) + k]);
                    }
                    //std::cout << "  ..." << std::endl;
                }
                // set display for lowercase dec numbers filled with ' '
                //std::cout << std::nouppercase << std::setfill(' ') << std::dec;

                // if pixel format starts with "Mono"
                if (std::string(pImage->GetPixelformat()).substr(0, 4) == "Mono") {
                    // transform to Mono8
                    pTransformImage = imgProcessor->CreateTransformedImage(pImage, "Mono8");
                    /*
                    std::cout << " Image "
                        << std::setw(5) << pBufferFilled->GetFrameID() << " transformed to Mono8" << std::endl;
                    std::cout << "  pTransformImage.Pixelformat:    "
                        << pTransformImage->GetPixelformat() << std::endl;
                    std::cout << "  pTransformImage.Width:          "
                        << pTransformImage->GetWidth() << std::endl;
                    std::cout << "  pTransformImage.Height:         "
                        << pTransformImage->GetHeight() << std::endl;
                    std::cout << "  pTransformImage.Buffer:         "
                        << std::hex << pTransformImage->GetBuffer() << std::dec << std::endl;
                    std::cout << "  Bytes per image:                "
                        << pTransformImage->GetWidth() * pTransformImage->GetHeight() * 1 << std::endl;
                    std::cout << "  Bytes per pixel:                "
                        << 1.0 << std::endl;
                    */
                    unsigned char* transformBuffer = (unsigned char*)pTransformImage->GetBuffer();

                    // display first 6 pixel values of first 6 lines of the transformed image
                    // ========================================================================
                    //std::cout << "  Address    Y  Y  Y  Y  Y  Y " << std::endl;

                    // set display for uppercase hex numbers filled with '0'
                    //std::cout << std::uppercase << std::setfill('0') << std::hex;
                    for (int j = 0; j < 6; j++) {  // first 6 lines
                        void* transformBufferAddress = &transformBuffer[pTransformImage->GetWidth() * 1 * j];
                        ///std::cout << "  " << std::setw(8) << std::setfill('0')
                         //   << std::hex << transformBufferAddress << " ";
                        for (int k = 0; k < 6; k++) {  // first 6 Pixel with Mono8 (1 Byte per Pixel)
                            // value of pixel
                           // std::cout << " " << std::setw(2)
                             //   << static_cast<int>(transformBuffer[pTransformImage->GetWidth() * j + k]);
                        }
                        //std::cout << " ..." << std::endl;
                    }
                    // set display for lowercase dec numbers filled with ' '
                    //std::cout << std::nouppercase << std::setfill(' ') << std::dec;
                    //std::cout << " " << std::endl;
                }
                else {  // if color format
                    // transform to BGR8
                    pTransformImage = imgProcessor->CreateTransformedImage(pImage, "BGR8");
                    /*
                    std::cout << " Image "
                        << std::setw(5) << pBufferFilled->GetFrameID() << " transformed to BGR8" << std::endl;
                    std::cout << "  pTransformImage.Pixelformat:    "
                        << pTransformImage->GetPixelformat() << std::endl;
                    std::cout << "  pTransformImage.Width:          "
                        << pTransformImage->GetWidth() << std::endl;
                    std::cout << "  pTransformImage.Height:         "
                        << pTransformImage->GetHeight() << std::endl;
                    std::cout << "  pTransformImage.Buffer:         "
                        << std::hex << pTransformImage->GetBuffer() << std::dec << std::endl;
                    std::cout << "  Bytes per image:                "
                        << pTransformImage->GetWidth() * pTransformImage->GetHeight() * 3 << std::endl;
                    std::cout << "  Bytes per pixel:                "
                        << 3.0 << std::endl;
                    */

                    unsigned char* transformBuffer = (unsigned char*)pTransformImage->GetBuffer();

                    // display first 6 pixel values of first 6 lines of the transformed image
                    // ========================================================================
                    //std::cout << "  Address    B  G  R  B  G  R  B  G  R  B  G  R  B  G  R  B  G  R" << std::endl;

                    // set display for uppercase hex numbers filled with '0'
                    std::cout << std::uppercase << std::setfill('0') << std::hex;
                    for (int j = 0; j < 6; j++) {  // 6 lines
                        void* transformBufferAddress = &transformBuffer[pTransformImage->GetWidth() * 3 * j];
                        std::cout << "  " << std::setw(8) << std::setfill('0')
                            << std::hex << transformBufferAddress << " ";
                        for (int k = 0; k < 6; k++) {  // first 6 Pixel with BGR8 (3 Bytes per Pixel)
                            // Value of Blue pixel
                            std::cout << " " << std::setw(2)
                                << static_cast<int>(transformBuffer[pTransformImage->GetWidth() * 3 * j + k * 3 + 0]);
                            // Value of Green pixel
                            std::cout << " " << std::setw(2)
                                << static_cast<int>(transformBuffer[pTransformImage->GetWidth() * 3 * j + k * 3 + 1]);
                            // Value of Red pixel
                            std::cout << " " << std::setw(2)
                                << static_cast<int>(transformBuffer[pTransformImage->GetWidth() * 3 * j + k * 3 + 2]);
                        }
                        std::cout << " ..." << std::endl;
                    }
                    // set display for lowercase dec numbers filled with ' '
                    std::cout << std::nouppercase << std::setfill(' ') << std::dec;
                    std::cout << " " << std::endl;
                }

                // OPEN CV STUFF
                m.lock();
                cam_img = cv::Mat(pTransformImage->GetHeight(), pTransformImage->GetWidth(), CV_8U, (int*)pTransformImage->GetBuffer());
                m.unlock();
                //display the current image in the window ----

                /*
                if (keypoints.size() == 0) {
                    cv::imshow("Camera", cam_img);
                }
                else {
                    cv::Mat drawn_img;
                    cv::drawKeypoints(cam_img, keypoints, drawn_img, cv::Scalar(255, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                    cv::imshow("Camera", drawn_img);
                }
                */
                std::vector<cv::KeyPoint> trap_locations;
                for (const auto& entry : spotManager->trapped_beads) {
                    // Extract x, y coordinates from the pair
                    int x = entry.first.second;
                    int y = entry.first.first;

                    // Create a cv::KeyPoint object
                    cv::KeyPoint keypoint(x, y, 20); // You might want to adjust the size parameter (third argument)

                    // Add the KeyPoint to the vector
                    trap_locations.push_back(keypoint);
                }

                cv::Mat imgWithKeypoints;
                cv::Mat imgWithTrapLocations;
                {
                    std::unique_lock<std::mutex> lock_m(m);
                    cv::drawKeypoints(cam_img, keypoints, imgWithKeypoints, cv::Scalar(255, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                }

                cv::drawKeypoints(imgWithKeypoints, trap_locations, imgWithTrapLocations, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                cv::imshow("Camera", imgWithTrapLocations);

                cv::waitKey(CAM_FRAME_RATE);

                pTransformImage->Release();
                // delete [] transformBuffer;

                // QUEUE BUFFER AFTER USE
                pBufferFilled->QueueBuffer();
            }
        }
        if (pImage != NULL) {
            pImage->Release();
            pImage = NULL;
        }
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }
    std::cout << " " << std::endl;


    std::cout << "CAMERA STOP" << std::endl;
    std::cout << "###########" << std::endl << std::endl;

    // STOP CAMERA
    try {
        // SEARCH FOR 'AcquisitionAbort'
        if (pDevice->GetRemoteNodeList()->GetNodePresent("AcquisitionAbort")) {
            pDevice->GetRemoteNode("AcquisitionAbort")->Execute();
            std::cout << "5.1.12   " << pDevice->GetModel() << " aborted " << std::endl;
        }

        pDevice->GetRemoteNode("AcquisitionStop")->Execute();
        std::cout << "5.1.12   " << pDevice->GetModel() << " stopped " << std::endl;
        std::cout << std::endl;

        BGAPI2::String sExposureNodeName = "";
        if (pDevice->GetRemoteNodeList()->GetNodePresent("ExposureTime")) {
            sExposureNodeName = "ExposureTime";
        }
        else if (pDevice->GetRemoteNodeList()->GetNodePresent("ExposureTimeAbs")) {
            sExposureNodeName = "ExposureTimeAbs";
        }
        std::cout << "         ExposureTime:                   "
            << std::fixed << std::setprecision(0) << pDevice->GetRemoteNode(sExposureNodeName)->GetDouble() << " ["
            << pDevice->GetRemoteNode(sExposureNodeName)->GetUnit() << "]" << std::endl;
        if (pDevice->GetTLType() == "GEV") {
            if (pDevice->GetRemoteNodeList()->GetNodePresent("DeviceStreamChannelPacketSize")) {
                std::cout << "         DeviceStreamChannelPacketSize:  "
                    << pDevice->GetRemoteNode("DeviceStreamChannelPacketSize")->GetInt() << " [bytes]" << std::endl;
            }
            else {
                std::cout << "         GevSCPSPacketSize:              "
                    << pDevice->GetRemoteNode("GevSCPSPacketSize")->GetInt() << " [bytes]" << std::endl;
            }
            std::cout << "         GevSCPD (PacketDelay):          "
                << pDevice->GetRemoteNode("GevSCPD")->GetInt() << " [tics]" << std::endl;
        }
        std::cout << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    // STOP DataStream acquisition
    try {
        if (pDataStream->GetTLType() == "GEV") {
            // DataStream Statistic
            std::cout << "         DataStream Statistics " << std::endl;
            std::cout << "           DataBlockComplete:              "
                << pDataStream->GetNodeList()->GetNode("DataBlockComplete")->GetInt() << std::endl;
            std::cout << "           DataBlockInComplete:            "
                << pDataStream->GetNodeList()->GetNode("DataBlockInComplete")->GetInt() << std::endl;
            std::cout << "           DataBlockMissing:               "
                << pDataStream->GetNodeList()->GetNode("DataBlockMissing")->GetInt() << std::endl;
            std::cout << "           PacketResendRequestSingle:      "
                << pDataStream->GetNodeList()->GetNode("PacketResendRequestSingle")->GetInt() << std::endl;
            std::cout << "           PacketResendRequestRange:       "
                << pDataStream->GetNodeList()->GetNode("PacketResendRequestRange")->GetInt() << std::endl;
            std::cout << "           PacketResendReceive:            "
                << pDataStream->GetNodeList()->GetNode("PacketResendReceive")->GetInt() << std::endl;
            std::cout << "           DataBlockDroppedBufferUnderrun: "
                << pDataStream->GetNodeList()->GetNode("DataBlockDroppedBufferUnderrun")->GetInt() << std::endl;
            std::cout << "           Bitrate:                        "
                << pDataStream->GetNodeList()->GetNode("Bitrate")->GetDouble() << std::endl;
            std::cout << "           Throughput:                     "
                << pDataStream->GetNodeList()->GetNode("Throughput")->GetDouble() << std::endl;
            std::cout << std::endl;
        }
        if (pDataStream->GetTLType() == "U3V") {
            // DataStream Statistic
            std::cout << "         DataStream Statistics " << std::endl;
            std::cout << "           GoodFrames:            "
                << pDataStream->GetNodeList()->GetNode("GoodFrames")->GetInt() << std::endl;
            std::cout << "           CorruptedFrames:       "
                << pDataStream->GetNodeList()->GetNode("CorruptedFrames")->GetInt() << std::endl;
            std::cout << "           LostFrames:            "
                << pDataStream->GetNodeList()->GetNode("LostFrames")->GetInt() << std::endl;
            std::cout << std::endl;
        }

        pDataStream->StopAcquisition();
        std::cout << "5.1.12   DataStream stopped " << std::endl;
        bufferList->DiscardAllBuffers();
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }
    std::cout << std::endl;


    std::cout << "RELEASE" << std::endl;
    std::cout << "#######" << std::endl << std::endl;

    // Release buffers
    std::cout << "5.1.13   Releasing the resources " << std::endl;
    try {
        while (bufferList->size() > 0) {
            pBuffer = *(bufferList->begin());
            bufferList->RevokeBuffer(pBuffer);
            delete pBuffer;
        }
        std::cout << "         buffers after revoke:    " << bufferList->size() << std::endl;

        pDataStream->Close();
        pDevice->Close();
        pInterface->Close();
        pSystem->Close();
        BGAPI2::SystemList::ReleaseInstance();
        // RELEASE IMAGE PROCESSOR
        delete imgProcessor;
        std::cout << "         ImageProcessor released" << std::endl << std::endl;
    }
    catch (BGAPI2::Exceptions::IException& ex) {
        returncode = (returncode == 0) ? 1 : returncode;
        std::cout << "ExceptionType:    " << ex.GetType() << std::endl;
        std::cout << "ErrorDescription: " << ex.GetErrorDescription() << std::endl;
        std::cout << "in function:      " << ex.GetFunctionName() << std::endl;
    }

    std::cout << std::endl;
    std::cout << "End" << std::endl << std::endl;

    std::cout << "Input any number to close the program:";
    int endKey = 0;
    std::cin >> endKey;
    return returncode;
}