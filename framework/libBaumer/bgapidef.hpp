/*****************************************************************************\
*    Program:		bgapi
*    Version:		$Revision: 8501 $	
*    Module:		$Source$
*    Compiler:		
*    Operating System:
*    Purpose:
*
*    Notes:
*    Created:		17.04.2007
*    Last modified:	$Date: 2009-10-28 12:06:25 +0100 (Mi, 28 Okt 2009) $
*                         
*                         
*    Author:		Silvio Voitzsch
*	 Last $Author: svo $
*    Copyright:		Baumer Optronic GmbH, Radeberg
*
*	 $Log: bgapidef.hpp,v $
*	 Revision 1.38  2008/04/08 08:52:12  svo
*	 io's and message channel
*	
*	 Revision 1.37  2008/04/07 07:03:03  svo
*	 io's and message channel
*	
*	 Revision 1.36  2008/04/07 07:00:23  svo
*	 *** empty log message ***
*	
*	 Revision 1.35  2008/04/07 06:16:47  svo
*	 io's and message channel
*	
*	 Revision 1.34  2008/03/13 10:02:33  svo
*	 new feature TRIGGERSYNCSW
*	
*	 Revision 1.33  2008/03/12 14:41:34  svo
*	 new bgapi feature DEFECTPIXELS
*	
*	 Revision 1.32  2008/03/06 15:00:47  oth
*	 added devicetriggermode
*	
*	 Revision 1.31  2008/03/03 14:56:51  svo
*	 creation of BGAPI_ImageHeader stucture
*	
*	 Revision 1.30  2008/02/22 13:15:15  svo
*	 BGAPI_ImageHeader defined
*	
*	 Revision 1.29  2008/02/22 12:07:00  oth
*	 added new Feature ToEMaster
*	
*	 Revision 1.28  2008/02/18 10:51:27  svo
*	 getReadoutTime added und basic feature id BGAPI_FEATUREID_CAMERA_READOUTTIME
*	
*	 Revision 1.27  2008/01/31 12:51:16  svo
*	 HDMI Format, Usersettinga and Warm pixel correction
*	
*	 Revision 1.26  2008/01/29 15:16:44  svo
*	 new camera feature monitorformat for hdmi
*	
*	 Revision 1.25  2008/01/21 07:39:40  svo
*	 new enumerators to disable scaling, filter and demosaic options
*	
*	 Revision 1.24  2008/01/03 15:42:16  svo
*	 New element 'hostnr' to camera device info structure added.
*	
*	 Revision 1.23  2007/12/12 10:54:24  svo
*	 *** empty log message ***
*	
*	 Revision 1.22  2007/12/11 12:33:47  oth
*	 new feature
*	
*	 Revision 1.21  2007/12/06 07:41:07  svo
*	 bo standard sequencer defined
*	
*	 Revision 1.20  2007/11/27 09:15:21  svo
*	 file header added
*	
\*****************************************************************************/

/** 
\file bgapidef.hpp
\brief Definitions for bgapi
*/


#ifndef _BGAPIDEF_HPP__
#define _BGAPIDEF_HPP__

#if !defined(_GNULINUX)
#define BGAPI_CALLBACK __stdcall
#else
#define BGAPI_CALLBACK
#endif

#include "bgapiresult.h"

typedef struct BGAPI_System BGAPI_System;
typedef struct BGAPI_Camera BGAPI_Camera;
typedef struct BGAPI_Image BGAPI_Image;
typedef struct BGAPI_Feature BGAPI_Feature;
typedef struct BGAPI_Value BGAPI_Value;
typedef struct BGAPI_TraceObject BGAPI_TraceObject;

namespace BGAPI
{
	class Value;
}

/**
\brief The enumeration BGAPI_FeatureId Defines all available features.
*/
enum BGAPI_FeatureId
{
	/** 
	\brief Special feature id. Use this feature id with function getFeature to get the state of all features at once.
	*/
	BGAPI_FEATUREID_ALL = 0,		
	
	/** 
	\brief Show a hardware specific control monitor.
	*/
    BGAPI_FEATUREID_SYSTEM_MONITOR = 101,
	
	/**
	\brief Get vendor informations: name, vendor id and devie type (see BGAPIX_DeviceType).
	*/
    BGAPI_FEATUREID_SYSTEM_DEVICEVENDORNAME = 102,

	/**
	\brief Get serial number of System.
	*/
	BGAPI_FEATUREID_SYSTEM_DEVICEID = 103,

	/**
	\brief Get used payload size for sending and reciving.
	*/
	BGAPI_FEATUREID_SYSTEM_DEVICEPAYLOAD = 104,
	
	/**
	\brief System Feature. Select the driver model for GigE Devices (see BGAPI_GVSDriverModel)
	*/
	BGAPI_FEATUREID_SYSTEM_GIGEDRIVERMODEL = 108,

	/**
	\brief System Feature. Force an IP for a special GigE Device (input is MAC address)
	*/
	BGAPI_FEATUREID_SYSTEM_GIGEFORCEIP = 109,

	/**
	\brief Version string. Shows the versions of used system module and drivers.
	*/
	BGAPI_FEATUREID_SYSTEM_VERSIONS = 110,

	/**
	\brief System Feature. Sends a command to a group of Cammeras
	*/
	BGAPI_FEATUREID_SYSTEM_COMMAND = 111,

	/**
	\brief System Feature. Sends a command to a group of Cammeras
	*/
	BGAPI_FEATUREID_SYSTEM_COMMANDRESPONDER = 112,

	/**
	\brief System Feature. Selects interfaces to use
	*/
	BGAPI_FEATUREID_SYSTEM_INTERFACES = 113,
	
	/**
	\brief Global trigger flag.
	*/
	BGAPI_FEATUREID_CAMERA_TRIGGER = 200,

	/**
	\brief Different sources for trigger input with a selection list (see BGAPI_TriggerSource).
	*/
	BGAPI_FEATUREID_CAMERA_TRIGGERSOURCE = 201,

	/**
	\brief Software trigger feature with an execute function.
	*/
	BGAPI_FEATUREID_CAMERA_TRIGGERSW = 202,

	/**
	\brief Trigger activation feature with a selection list (see definition of BGAPI_Activation).
	*/
	BGAPI_FEATUREID_CAMERA_TRIGGERACTIVATION = 203,

	/**
	\brief Delay for trigger start. 0 ... n events.
	*/
    BGAPI_FEATUREID_CAMERA_TRIGGERDELAY = 212,

	/**
	\brief Line synchronisation for line cameras.
	*/
	BGAPI_FEATUREID_CAMERA_TRIGGER_HW_LINESYNC = 213,    

	/**
	\brief Activates a broadcast software trigger command.
	*/
	BGAPI_FEATUREID_CAMERA_TRIGGERSYNCSW = 214,

	/**
	\brief Flash mode feature: with on/off switch.
	*/
	BGAPI_FEATUREID_CAMERA_FLASH = 220, 

	/**
	\brief Activation of flash with a selection list (see BGAPI_Activation).
	*/
	BGAPI_FEATUREID_CAMERA_FLASHACTIVATION = 225,
	
	/**
	\brief Delay for flash start. 0 ... n events.
	*/
	BGAPI_FEATUREID_CAMERA_FLASHDELAY = 226,

	/**
	\brief Delay for transmission start. 0 ... n events.
	*/
	BGAPI_FEATUREID_CAMERA_TRANSMISSIONDELAY = 227,

	/**
	\brief Digital parallel port. Query int value 32 Bit.
	*/
	BGAPI_FEATUREID_CAMERA_DIGPPORT = 230,

	/**
	\brief Digital serial port. Query string.
	*/
	BGAPI_FEATUREID_CAMERA_DIGSPORT = 232,

	/**
	\brief Digital routed output.
	*/
	BGAPI_FEATUREID_CAMERA_DIGROUTPUT = 234,

	/**
	\brief Analog global gain.
	*/
	BGAPI_FEATUREID_CAMERA_AGAIN = 235,

	/**
	\brief Analog global offset.
	*/
    BGAPI_FEATUREID_CAMERA_AOFFSET = 236,

	/**
	\brief Digital global gain.
	*/
	BGAPI_FEATUREID_CAMERA_DGAIN = 237,

	/**
	\brief Exposure value.
	*/
	BGAPI_FEATUREID_CAMERA_EXPOSURE = 239,

	/**
	\brief Shutter mode feature with a selection list (see BGAPI_Shutter).
	*/
	BGAPI_FEATUREID_CAMERA_SHUTTER = 240,

	/**
	\brief Exposure can be automatically adjusted. Feature with on/off switch.
	*/
	BGAPI_FEATUREID_CAMERA_AUTOEXPOSURE = 241,

	/**
	\brief Current height of the image/ROI in pixel.
	*/
	BGAPI_FEATUREID_CAMERA_HEIGHT = 248,
	
	/**
	\brief Current width of the image/ROI in pixel.
	*/
	BGAPI_FEATUREID_CAMERA_WIDTH = 249,
	
	/**
	\brief Partial scan with on/off switch and ROI.
	*/
	BGAPI_FEATUREID_CAMERA_PARTIALSCAN = 250,

	/**
	\brief Starts/Stops readout of sensor and the data transfer. Feature with on/off switch.
	*/
    BGAPI_FEATUREID_CAMERA_START = 251,

	/**
	\brief Test pattern with on/off switch and  with a selection list (see BGAPI_TestPattern).
	*/
	BGAPI_FEATUREID_CAMERA_TESTPATTERN = 252,

	/**
	\brief Cooler function, with on/off switch.
	*/
    BGAPI_FEATUREID_CAMERA_COOLER = 253,

	/**
	\brief Reset function as one push execution. Loads the factory settings.
	*/
    BGAPI_FEATUREID_CAMERA_RESET = 254,

	/**
	\brief Asynchronous reset function as one push execution.
	*/
	BGAPI_FEATUREID_CAMERA_RESETASYNC = 255,

	/**
	\brief Read out mode with a selection list (see BGAPI_Readout).
	*/
	BGAPI_FEATUREID_CAMERA_READOUT = 256,

	/**
	\brief Look up table.
	*/
	BGAPI_FEATUREID_CAMERA_LOOKUPTABLE_GAMMA = 260,

	/**
	\brief Set Whitebalance factors.
	*/
	BGAPI_FEATUREID_CAMERA_WHITEBALANCE_SET = 261,

	/**
	\brief Whitebalance option as one push execution.
	*/
	BGAPI_FEATUREID_CAMERA_WHITEBALANCE_DO = 262,

	/**
	\brief Internal command to camera.
	*/
	BGAPI_FEATUREID_CAMERA_COMMAND = 263,

	/**
	\brief Image format option with a selection list (e.g. 1024x768, 640x480, ...).
	*/
    BGAPI_FEATUREID_CAMERA_IMAGEFORMAT = 264,

	/**
	\brief Image code option with a selection list (e.g. raw, yuv, rgb, ...).
	*/
	BGAPI_FEATUREID_CAMERA_IMAGEFORMAT_CODE = 265,

	/**
	\brief Set packet size for image transfer.
	*/
	BGAPI_FEATUREID_CAMERA_IMAGEFORMAT_PACKETSIZE = 266,

	/**
	\brief Set frame rate in frames per seconds for a DCAM camera.
	*/
	BGAPI_FEATUREID_CAMERA_IMAGEFORMAT_FPS = 267,

	/**
	\brief Set frame rate in frames per seconds.
	*/
	BGAPI_FEATUREID_CAMERA_FPS = 268,

	/**
	\brief Select a temperature sensor.
	*/
	BGAPI_FEATUREID_CAMERA_TEMPERATURESENSORSELECTOR = 274,

	/**
	\brief Query the actual temperature of current selected temperature sensor.
	*/
	BGAPI_FEATUREID_CAMERA_TEMPERATURESENSOR = 275,

	/**
	\brief Set the critical temperature of current selected temperature sensor.
	*/
	BGAPI_FEATUREID_CAMERA_TEMPERATURESENSORCRITICAL = 276,

	/**
	\brief Set the scale factor in x direction.
	*/
	BGAPI_FEATUREID_CAMERA_SCALINGX = 280,

	/**
	\brief Set the scale factor in y direction.
	*/
	BGAPI_FEATUREID_CAMERA_SCALINGY = 281,

	/**
	\brief Set the gain factor in 'L' direction for Lab color space.
	*/
	BGAPI_FEATUREID_CAMERA_GAINL = 282,

	/**
	\brief Set the gain factor in 'a' direction for Lab color space.
	*/
	BGAPI_FEATUREID_CAMERA_GAINA = 283,

	/**
	\brief Set the gain factor in 'b' direction for Lab color space.
	*/
	BGAPI_FEATUREID_CAMERA_GAINB = 284,

	/**
	\brief Set the offset in 'L' direction for Lab color space.
	*/
	BGAPI_FEATUREID_CAMERA_OFFSETL = 285,

	/**
	\brief Set the offset in 'a' direction for Lab color space.
	*/
	BGAPI_FEATUREID_CAMERA_OFFSETA = 286,

	/**
	\brief Set the offset in 'b' direction for Lab color space.
	*/
	BGAPI_FEATUREID_CAMERA_OFFSETB = 287,
	
	/**
	\brief Access mode to image data with a selection list (see definition of BGAPI_DataAccess).
	*/
	BGAPI_FEATUREID_CAMERA_DATAACCESS = 300,

	/**
	\brief Frame counter option.
	*/
	BGAPI_FEATUREID_CAMERA_FRAMECOUNTER = 301,

	/**
	\brief Select camera canal.
	*/
	BGAPI_FEATUREID_CAMERA_CANAL = 302,

	/**
	\brief Brightness correction for binning based image formats.
	*/
	BGAPI_FEATUREID_CAMERA_BRIGHTNESSCORRECTION = 303,

	/**
	\brief Horizontal image configuration for line cameras.
	*/	
	BGAPI_FEATUREID_CAMERA_HORIZONTALAQUIRE = 330,

	/**
	\brief Vertical image configuration for line cameras.
	*/	
	BGAPI_FEATUREID_CAMERA_VERTICALAQUIRE = 331,

	/**
	\brief Vertical image configuration for line cameras.
	*/	
	BGAPI_FEATUREID_CAMERA_STATISTIC = 332,

	/**
	\brief Hardware supported sequnce of image aquisition. Use this feature to enable the sequencer.
	*/	
	BGAPI_FEATUREID_CAMERA_BOSEQUENCERENABLE = 333,

	/**
	\brief Hardware supported sequnce of image aquisition. Use this feature to start the sequencer.
	*/	
	BGAPI_FEATUREID_CAMERA_BOSEQUENCERSTART = 334,

	/**
	\brief Hardware supported sequnce of image aquisition. Use this function to select the run mode of sequencer.
	*/	
	BGAPI_FEATUREID_CAMERA_BOSEQUENCERRUNMODE = 335,

	/**
	\brief Hardware supported sequnce of image aquisition. Use this feature to add a new parameter set.
	*/	
	BGAPI_FEATUREID_CAMERA_BOSEQUENCERADD = 336,

	/**
	\brief Hardware supported sequnce of image aquisition. Use this feature to reset parameter set of sequencer.
	*/	
	BGAPI_FEATUREID_CAMERA_BOSEQUENCERRESET = 337,

	/**
	\brief Hardware supported sequnce of image aquisition. Use this function to get the current sequence counters.
	*/	
	BGAPI_FEATUREID_CAMERA_BOSEQUENCERCOUNTER = 338,

	/**
	\brief Get the logical host number with the camera is connected. Valid for Device types BGAPIX_DEVICETYPE_BAUMERCAM_FCAMX, BGAPIX_DEVICETYPE_BAUMERCAM_DCAM and BGAPIX_DEVICETYPE_BAUMERCAM_GIGE.
	*/
	BGAPI_FEATUREID_CAMERA_DEVICEHOSTNR = 339,

	/**
	\brief Readout time of camera in ms.
	*/
	BGAPI_FEATUREID_CAMERA_READOUTTIME = 340,

	/**
	\brief Returns defect pixels of camera.
	*/
	BGAPI_FEATUREID_CAMERA_DEFECTPIXELS = 341,

	/**
	\brief Enables the correctionof defect pixels of the camera.
	*/
	BGAPI_FEATUREID_CAMERA_DEFECTPIXELSENABLE = 342,

	/**
	\brief BGAPI_FEATUREID_CAMERA_LINESELECTOR
	*/
	BGAPI_FEATUREID_CAMERA_LINESELECTOR = 343,

	/**
	\brief BGAPI_FEATUREID_CAMERA_LINEMODE
	*/
	BGAPI_FEATUREID_CAMERA_LINEMODE = 344,

	/**
	\brief BGAPI_FEATUREID_CAMERA_LINEDEBOUNCERHIGH
	*/
	BGAPI_FEATUREID_CAMERA_LINEDEBOUNCERHIGH = 345,

	/**
	\brief BGAPI_FEATUREID_CAMERA_LINEDEBOUNCERLOW
	*/
	BGAPI_FEATUREID_CAMERA_LINEDEBOUNCERLOW = 346,

	/**
	\brief BGAPI_FEATUREID_CAMERA_LINESTATE
	*/
	BGAPI_FEATUREID_CAMERA_LINESTATE = 347,

	/**
	\brief BGAPI_FEATUREID_CAMERA_LINESTATEALL
	*/
	BGAPI_FEATUREID_CAMERA_LINESTATEALL = 348,

	/**
	\brief BGAPI_FEATUREID_CAMERA_LINEOUTPUTSOURCE
	*/
	BGAPI_FEATUREID_CAMERA_LINEOUTPUTSOURCE = 349,

	/**
	\brief BGAPI_FEATUREID_CAMERA_USEROUTPUTSELECTOR
	*/
	BGAPI_FEATUREID_CAMERA_USEROUTPUTSELECTOR = 350,

	/**
	\brief BGAPI_FEATUREID_CAMERA_USEROUTPUTVALUE
	*/
	BGAPI_FEATUREID_CAMERA_USEROUTPUTVALUE = 351,

	/**
	\brief BGAPI_FEATUREID_CAMERA_USEROUTPUTVALUEALL
	*/
	BGAPI_FEATUREID_CAMERA_USEROUTPUTVALUEALL = 352,

	/**
	\brief Select the message id's for asynchronous message channel output. This feature also returns all available message id's.
	*/
	BGAPI_FEATUREID_CAMERA_MESSAGEID = 353,

	/**
	\brief This feature enable a message which should be signald through the message channel.
	*/
	BGAPI_FEATUREID_CAMERA_MESSAGESTATE = 354,

	/**
	\brief Select the desired look up table.
	*/
	BGAPI_FEATUREID_CAMERA_LOOKUPTABLESELECTOR = 359,

	/**
	\brief Enable the selected look up table.
	*/
	BGAPI_FEATUREID_CAMERA_LOOKUPTABLEENABLE = 360,

	/**
	\brief Manipulate the look up table.
	*/
	BGAPI_FEATUREID_CAMERA_LOOKUPTABLE = 361,

	/**
	\brief Set the gamma value. This feature may change the look up table in the camera.
	*/
	BGAPI_FEATUREID_CAMERA_GAMMA = 362,

	/**
	\brief Set the line potential.
	*/
	BGAPI_FEATUREID_CAMERA_LINEPOTENTIAL = 363,
	
	/**
	\brief Set the distance amplitude threshold.
	*/
	BGAPI_FEATUREID_CAMERA_DISTAMPLITUDETHRESHOLD = 364,

	/**
	\brief Enable the camera/trigger device config mode.
	*/
	BGAPI_FEATUREID_CAMERA_CONFIGMODE = 365,

	/**
	\brief Set the activation state of an input line.
	*/
	BGAPI_FEATUREID_CAMERA_LINEACTIVATION = 366,

	/**
	\brief Set the delay time in ticks of an input line.
	*/
	BGAPI_FEATUREID_CAMERA_LINEDELAY = 367,

	/**
	\brief Set the timebase of delay time.
	*/
	BGAPI_FEATUREID_CAMERA_LINEDELAYTIMEBASE = 368,

	/**
	\brief Set the clockrate divider of timebase.
	*/
	BGAPI_FEATUREID_CAMERA_LINEDELAYCLOCKRATEDIVIDER = 369,

	/**
	\brief The supported action commands of an action command receiver.
	*/
	BGAPI_FEATUREID_CAMERA_RECEIVERACTIONCOMMAND = 370,

	/**
	\brief Defines the device key for an action command receiver.
	*/
	BGAPI_FEATUREID_CAMERA_RECEIVERDEVICEKEY = 371,

	/**
	\brief Defines the camera group for an action command receiver.
	*/
	BGAPI_FEATUREID_CAMERA_RECEIVERGROUP = 372,
	/**
	\brief The supported action commands of an action command sender.
	*/
	BGAPI_FEATUREID_CAMERA_SENDERACTIONCOMMAND = 373,

	/**
	\brief Defines the device key for an action command sender.
	*/
	BGAPI_FEATUREID_CAMERA_SENDERDEVICEKEY = 374,

	/**
	\brief Defines the camera group for an action command sender.
	*/
	BGAPI_FEATUREID_CAMERA_SENDERGROUP = 375,

	/**
	\brief Offers a list with responders for every defined action command.
	*/
	BGAPI_FEATUREID_CAMERA_SENDERCOMMANDRESPONDER = 376,
	
	/**
	\brief Defines a timeout of an action command sender.
	*/
	BGAPI_FEATUREID_CAMERA_SENDERTIMEOUT = 377,

	/**
	\brief Defines a retry value of an action command sender.
	*/
	BGAPI_FEATUREID_CAMERA_SENDERRETRY = 378,

	/**
	\brief Set the pulse delay in ticks of an output line.
	*/
	BGAPI_FEATUREID_CAMERA_LINEPULSEDELAY = 379,

	/**
	\brief Set the timebase of delay time.
	*/
	BGAPI_FEATUREID_CAMERA_LINEPULSEDELAYTIMEBASE = 380,

	/**
	\brief Set the clockrate divider of timebase.
	*/
	BGAPI_FEATUREID_CAMERA_LINEPULSEDELAYCLOCKRATEDIVIDER = 381,

	/**
	\brief Set the pulse width of an output line.
	*/
	BGAPI_FEATUREID_CAMERA_LINEPULSEWIDTH = 382,

	/**
	\brief Set the timebase of pulse width.
	*/
	BGAPI_FEATUREID_CAMERA_LINEPULSEWIDTHTIMEBASE = 383,

	/**
	\brief Set the clockrate divider of pulse width timebase.
	*/
	BGAPI_FEATUREID_CAMERA_LINEPULSEWIDTHCLOCKRATEDIVIDER = 384,

	/**
	\brief Control the line output gate.
	*/
	BGAPI_FEATUREID_CAMERA_LINEGATE = 385,

	/**
	\brief Set the retrigger time. Only when using the trigger source BGAPI_TRIGGERSOURCE_RETRIGGER.
	*/
	BGAPI_FEATUREID_CAMERA_LINERETRIGGERTIME = 386,

	/**
	\brief Set the timebase of retrigger time.
	*/
	BGAPI_FEATUREID_CAMERA_LINERETRIGGERTIMEBASE = 387,

	/**
	\brief Set the clockrate divider of retrigger time.
	*/
	BGAPI_FEATUREID_CAMERA_LINERETRIGGERCLOCKRATEDIVIDER = 388,

	/**
	\brief Invert a output line.
	*/
	BGAPI_FEATUREID_CAMERA_LINEINVERTER = 389,

	/**
	\brief Camera supported horizontal flipping.
	*/
	BGAPI_FEATUREID_CAMERA_FLIPX = 390,
	
	/**
	\brief Camera supported vertical flipping.
	*/
	BGAPI_FEATUREID_CAMERA_FLIPY = 391,

	/**
	\brief Auto exposure support.
	*/
	BGAPI_FEATUREID_CAMERA_EXPOSUREAUTOMODE = 392,

	/**
	\brief Auto exposure support.
	*/
	BGAPI_FEATUREID_CAMERA_EXPOSUREAUTOBRIGHT = 393,

	/**
	\brief Auto exposure support.
	*/
	BGAPI_FEATUREID_CAMERA_EXPOSUREAUTOROI = 394,

	/**
	\brief Auto exposure support.
	*/
	BGAPI_FEATUREID_CAMERA_EXPOSUREAUTOMIN = 395,

	/**
	\brief Auto exposure support.
	*/
	BGAPI_FEATUREID_CAMERA_EXPOSUREAUTOMAX = 396,

	/**
	\brief Auto gain support.
	*/
	BGAPI_FEATUREID_CAMERA_GAINAUTOMODE = 397,

	/**
	\brief Auto gain support.
	*/
	BGAPI_FEATUREID_CAMERA_GAINAUTOBRIGHT = 398,

	/**
	\brief Auto gain support.
	*/
	BGAPI_FEATUREID_CAMERA_GAINAUTOROI = 399,

	/**
	\brief Auto gain support.
	*/
	BGAPI_FEATUREID_CAMERA_GAINAUTOMIN = 400,

	/**
	\brief Auto gain support.
	*/
	BGAPI_FEATUREID_CAMERA_GAINAUTOMAX = 401,

	/**
	\brief HDR support.
	*/
	BGAPI_FEATUREID_CAMERA_HDRENABLE = 402,

	/**
	\brief HDR configuration.
	*/
	BGAPI_FEATUREID_CAMERA_HDRIMAGESELECTOR = 403,
	
	/**
	\brief HDR configuration.
	*/
	BGAPI_FEATUREID_CAMERA_HDRPARAMETER = 404,
	
	/**
	\brief HDR configuration.
	*/
	BGAPI_FEATUREID_CAMERA_HDRPARAMETERSELECTOR = 405,
	
	/**
	\brief HDR configuration.
	*/
	BGAPI_FEATUREID_CAMERA_HDRNUMBEROFIMAGES = 406,

	/**
	\brief ToF Modulation frequency support.
	*/
	BGAPI_FEATUREID_CAMERA_TOFMODULATIONFREQUENCY = 415,
	
	/**
	\brief Select camera filter operations.
	*/
	BGAPI_FEATUREID_CAMERA_FILTERSELECTOR = 418,

	/**
	\brief Enable camera filter operations.
	*/
	BGAPI_FEATUREID_CAMERA_FILTER = 419,

	/**
	\brief Timer
	*/
	BGAPI_FEATUREID_CAMERA_TIMERSELECTOR = 420,

	/**
	\brief Timer
	*/
	BGAPI_FEATUREID_CAMERA_TIMERDURATION = 421,
	
	/**
	\brief Timer
	*/
	BGAPI_FEATUREID_CAMERA_TIMERDELAY = 422,
	
	/**
	\brief Timer
	*/
	BGAPI_FEATUREID_CAMERA_TIMERTRIGGERSOURCE = 423,
	
	/**
	\brief Timer
	*/
	BGAPI_FEATUREID_CAMERA_TIMERTRIGGERACTIVATION = 424,
	
	/**
	\brief Counter
	*/
	BGAPI_FEATUREID_CAMERA_COUNTERSELECTOR = 430,
	
	/**
	\brief Counter
	*/
	BGAPI_FEATUREID_CAMERA_COUNTERTRIGGERSOURCE = 431,
	
	/**
	\brief Counter
	*/
	BGAPI_FEATUREID_CAMERA_COUNTERTRIGGERACTIVATION = 432,
	
	/**
	\brief Counter
	*/
	BGAPI_FEATUREID_CAMERA_COUNTERVALUE = 433,
	
	/**
	\brief Load image from file.
	*/	
	BGAPI_FEATUREID_IMAGE_FILELOAD = 480,

	/**
	\brief Internal feature id is a help id, which is used for all get functions of the fundamental image conversion. Attention, this feature id can not be used with Basic API functions. 
	*/
	BGAPI_FEATUREID_IMAGE_INTERNAL = 481,

	/**
	\brief String VendorName
	*/
	BGAPI_FEATUREID_CAMERA_DEVICEVENDORNAME = 500,

	/**
	\brief String ModelName
	*/
	BGAPI_FEATUREID_CAMERA_DEVICEMODELNAME = 501,

	/**
	\brief Model identifier.
	*/
	BGAPI_FEATUREID_CAMERA_DEVICEMODELID = 502,
	
	/**
	\brief String additional info about device
	*/
	BGAPI_FEATUREID_CAMERA_DEVICEMANUFACTORINFO = 503,

	/**
	\brief String Device version
	*/
	BGAPI_FEATUREID_CAMERA_DEVICEVERSION = 504,

	/**
	\brief String firmware/software version
	*/
	BGAPI_FEATUREID_CAMERA_DEVICEFIRMWAREVERSION = 505,

	/**
	\brief String Serial Number
	*/
	BGAPI_FEATUREID_CAMERA_DEVICEID = 506,

	/**
	\brief String userprogrammable ID
	*/
	BGAPI_FEATUREID_CAMERA_DEVICEUSERID = 507,

	/**
	\brief Enum Scantype: Areascan, Linescan
	*/
	BGAPI_FEATUREID_CAMERA_DEVICESCANTYPE = 508,

	/**
	\brief Enum Type of sensor data: Mono/Color
	*/
	BGAPI_FEATUREID_CAMERA_DEVICESENSORDATA = 509,
	
	/**
	\brief HDMI support. Set HDMI Monitor Format.
	*/
	BGAPI_FEATUREID_CAMERA_MONITORFORMAT = 510,
	
	/**
	\brief Reload your own user settings from camera.
	*/
	BGAPI_FEATUREID_CAMERA_USERSETTINGS_SELECTOR = 511,

	/**
	\brief Save current camera settings as user settings.
	*/
	BGAPI_FEATUREID_CAMERA_USERSETTINGS_RELOAD = 512,

	/**
	\brief Save current camera settings as user settings.
	*/
	BGAPI_FEATUREID_CAMERA_USERSETTINGS_STORE = 513,
	
	/**
	\brief Save current camera settings as user settings.
	*/
	BGAPI_FEATUREID_CAMERA_USERSETTINGS_DEFAULTSELECTOR = 514,
	
	/**
	\brief Delete the user settings.
	*/
	BGAPI_FEATUREID_CAMERA_USERSETTINGS_RESET = 515,

	/**
	\brief Control the warm pixel correction.
	*/
	BGAPI_FEATUREID_CAMERA_WARMPIXELCORRECTION = 516,

	/**
	\brief Control the dark pixel correction.
	*/
	BGAPI_FEATUREID_CAMERA_DARKPIXELCORRECTION = 517,

	/**
	\brief Control the defect pixel correction.
	*/
	BGAPI_FEATUREID_CAMERA_DEFECTPIXELCORRECTION = 518,

	/**
	\brief PWM selector.
	*/
	BGAPI_FEATUREID_CAMERA_PWMSELECTOR = 519,

	/**
	\brief PWM Enable the selected PWM Signal.
	*/
	BGAPI_FEATUREID_CAMERA_PWMENABLE = 520,

	/**
	\brief PWM frequency.
	*/
	BGAPI_FEATUREID_CAMERA_PWMFREQUENCY = 521,

	/**
	\brief PWM duty cycle.
	*/
	BGAPI_FEATUREID_CAMERA_PWMDUTYCYCLE = 522,

	/**
	\brief PWM over power threshold.
	*/
	BGAPI_FEATUREID_CAMERA_PWMOVERPOWERTHRESHOLD = 523,
	
	/**
	\brief  Hardware supported sequnce of image aquisition. Use this feature to set again for current parameter set.
	*/
	BGAPI_FEATUREID_CAMERA_BOSEQUENCERAGAIN = 524,
	
	/**
	\brief Hardware supported sequnce of image aquisition. Use this feature to set exposure for current parameter set.
	*/
	BGAPI_FEATUREID_CAMERA_BOSEQUENCEREXPOSURE = 525,
	
	/**
	\brief PWM Hardware supported sequnce of image aquisition. Use this feature to set pwm duty cycle for current parameter set.
	*/
	BGAPI_FEATUREID_CAMERA_BOSEQUENCERPWMDUTYCYCLE = 526,

	/**
	\brief Selector for chunk data.
	*/
	BGAPI_FEATUREID_CAMERA_CHUNKSELECTOR = 527,

	/**
	\brief Enables chunk data.
	*/
	BGAPI_FEATUREID_CAMERA_CHUNKENABLE = 528,

	/**
	\brief Enables the chunk mode.
	*/
	BGAPI_FEATUREID_CAMERA_CHUNKMODEENABLE = 529,

	/**
	\brief Return the camera config file.
	*/
	BGAPI_FEATUREID_CAMERA_CONFIGFILE = 530,

	/**
	\brief Return the current data length.
	*/
	BGAPI_FEATUREID_CAMERA_DATALENGTH = 531,

	/**Start of GigE specific ID's at 1000
	*/

	/**
	\brief Maximum transmission unit. Only supported by BGAPIX_DEVICETYPE_BAUMERCAM_GIGE.
	*/
	BGAPI_FEATUREID_CAMERA_GIGEDEVICEMTU = 1010,

	/** M
	\brief CMD starts acquisation
	*/
	BGAPI_FEATUREID_CAMERA_GIGEACQUISITIONSTART = 1102,
	
	/** M
	\brief CMD stops acquisation
	*/
	BGAPI_FEATUREID_CAMERA_GIGEACQUISITIONSTOP = 1103,
		
	/** Baumer
	\brief CMD Pause the transmission.
	*/
	BGAPI_FEATUREID_CAMERA_GIGEINTERFACEPAUSE = 1148,
		
	/** Baumer
	\brief CMD Resumes the transmission.
	*/
	BGAPI_FEATUREID_CAMERA_GIGEINTERFACERESUME = 1149,
	
	
	/**
	\brief Bool Send additional Chunkdata to the image
	*/
	BGAPI_FEATUREID_CAMERA_GIGEUSEINFOHEADER = 1198,

	/**
	\brief Bool show the HWC in the first Bytes of the image
	*/
	BGAPI_FEATUREID_CAMERA_GIGESHOWINFOHEADERINIMAGE = 1199,
	
	/**O
	\brief Int MAC Adress of current Interface
	*/
	BGAPI_FEATUREID_CAMERA_GIGEMACADDRESS = 1407,

	/**O
	\brief BOOL Indicate if DHCP is supported by current Interface
	*/
	BGAPI_FEATUREID_CAMERA_GIGEIPCONFIGURATIONDHCP = 1409,

	/**O
	\brief Int IP Address of current Interface
	*/
	BGAPI_FEATUREID_CAMERA_GIGECURRENTIPADDRESS = 1412,

	/**O
	\brief Int Persistent IP Address of current Interface
	*/
	BGAPI_FEATUREID_CAMERA_GIGEPERSISTENTIPADDRESS = 1418,

	/**O
	\brief Int current Heartbeattimeout in ms
	*/
	BGAPI_FEATUREID_CAMERA_GIGEHEARTBEATTIMEOUT = 1428,

	/**O
	\brief Int Represents the the latched Value (from timestampvalueregister)
	*/
	BGAPI_FEATUREID_CAMERA_GIGETIMESTAMP = 1432,

	/**Baumer
	\brief Sets the Framecounter and executes a softwaretrigger
	*/
	BGAPI_FEATUREID_CAMERA_MARKTRIGGERIMAGE = 1444,

	/**Baumer
	\brief Read or write Cam registers
	*/
	BGAPI_FEATUREID_CAMERA_GIGEREGACCESS = 1445,

	/**Baumer
	\brief Int max time between two streaming packets in ms
	*/
	BGAPI_FEATUREID_CAMERA_GIGERESENDTIMEOUT = 1446,

	/**Baumer
	\brief Bool Restart the pic and fpga
	*/
	BGAPI_FEATUREID_CAMERA_REBOOT = 1447,

	/**Baumer
	\brief Int timedelay between two streaming packets
	*/
	BGAPI_FEATUREID_CAMERA_GIGESTREAMCHANNELPACKETDELAY = 1448,
		
	/**Baumer
	\brief Bool Reset the CCP Value
	*/
	BGAPI_FEATUREID_CAMERA_GIGECCPOVERRIDE = 1449,
	
	/**Baumer
	\brief Special Gige Triggerdevice Mode 
	*/
	BGAPI_FEATUREID_CAMERA_GIGETRIGGERDEVICEMODE = 1493,
	
	/**Baumer
	\brief Special Gige Multicast
	*/
	BGAPI_FEATUREID_CAMERA_GIGEMULTICAST = 1494,

	/**Baumer
	\brief Special Gige Multicast
	*/
	BGAPI_FEATUREID_CAMERA_GIGEMESSAGEMULTICAST = 1495,

	/**Baumer
	\brief Special Gige disable Heartbeat
	*/
	BGAPI_FEATUREID_CAMERA_GIGEENABLEHEARTBEAT = 1496,

	/*
	< reserved area for GigE specific Id's End = 1600 ==> next usable ID is 1601 */
    
	/* reserved area for oem specific Id's Start 4096 
	*/
	
	/**
	\brief OEM feature. Not documented here. See oem help for detailed description.
	*/
	BGAPI_FEATUREID_CAMERA_SEQUENCER = 4096,
	

	/**
	\brief OEM feature. Not documented here. See special oem help for detailed description.
	*/
	BGAPI_FEATUREID_CAMERA_SEQUENCERCONFIG = 4097,

	/*
	< reserved area for oem specific Id's End = 8192 */

	/**
	\brief Don't use that.
	*/	
    BGAPI_FEATUREID_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_ValueId defines the value importance.
*/
enum BGAPI_ValueId
{

	/**
	\brief Default value of value ID.
	*/	
	BGAPI_VALUEID_DEFAULT = 0,

	/**
	\brief The value represents a name.
	*/	
	BGAPI_VALUEID_NAME = 1,

	/**
	\brief The value represents a enable/disable functionality.
	*/	
	BGAPI_VALUEID_ONOFF = 2,

	/**
	\brief The value represents the top of a ROI.
	*/	
	BGAPI_VALUEID_TOP = 3,

	/**
	\brief The value represents the left of a ROI.
	*/	
    BGAPI_VALUEID_LEFT = 4,

	/**
	\brief The value represents the right of a ROI.
	*/	
	BGAPI_VALUEID_RIGHT = 5,

	/**
	\brief The value represents the bottom of a ROI.
	*/	
	BGAPI_VALUEID_BOTTOM = 6,

	/**
	\brief The value represents a offset in x direction.
	*/	
	BGAPI_VALUEID_OFFX = 7,

	/**
	\brief The value represents a offset in y direction.
	*/	
	BGAPI_VALUEID_OFFY = 8,

	/**
	\brief The value represents a size in x direction.
	*/	
    BGAPI_VALUEID_SIZEX = 9,

	/**
	\brief The value represents a size in y direction.
	*/	
	BGAPI_VALUEID_SIZEY= 10,

	/**
	\brief The value represents a width.
	*/	
	BGAPI_VALUEID_WIDTH = 11,

	/**
	\brief The value represents a height.
	*/	
	BGAPI_VALUEID_HEIGHT = 12,

	/**
	\brief The value represents an index.
	*/	
	BGAPI_VALUEID_INDEX = 13,

	/**
	\brief The value represents an indentifer.
	*/	
	BGAPI_VALUEID_IDENTIFIER = 14,

	/**
	\brief The value represents an enumerator.
	*/	
	BGAPI_VALUEID_ENUMERATOR= 15,

	/**
	\brief The value represents bytes per pixel.
	*/	
	BGAPI_VALUEID_BYTESPERPIXEL = 16,

	/**
	\brief The value represents bits per pixel.
	*/	
	BGAPI_VALUEID_BITSPERPIXEL = 17,

	/**
	\brief The value represents canals.
	*/	
	BGAPI_VALUEID_CANALS = 18,

	/**
	\brief The value represents planes.
	*/	
	BGAPI_VALUEID_PLANES = 19,

	/**
	\brief The value represents a read mask.
	*/	
	BGAPI_VALUEID_MASKREAD = 20,

	/**
	\brief The value represents a write mask.
	*/	
	BGAPI_VALUEID_MASKWRITE = 21,

	/**
	\brief The value represents a target value (e.g. for auto exposure).
	*/	
	BGAPI_VALUEID_TARGET = 22,

	/**
	\brief The value represents a one push execution (e.g. doTrigger, doWhiteBalance).
	*/	
	BGAPI_VALUEID_EXECUTE = 23,

	/**
	\brief The value represents the canal 0 (e.g. color canal, cyan, red or mono).
	*/	
	BGAPI_VALUEID_CANAL0 = 24, 

	/**
	\brief The value represents the canal 1 (e.g. color canal yellow, green).
	*/	
	BGAPI_VALUEID_CANAL1 = 25, 

	/**
	\brief The value represents the canal 2 (e.g. color canal magenta, blue).
	*/	
	BGAPI_VALUEID_CANAL2 = 26,

	/**
	\brief The value represents the canal 3 (e.g. color canal black, alpha canal)
	*/	
	BGAPI_VALUEID_CANAL3 = 27,

	/**
	\brief The value represents a scale factor in x direction.
	*/	
	BGAPI_VALUEID_SCALEFACTORX = 28,

	/**
	\brief The value represents a scale factor in y direction.
	*/	
	BGAPI_VALUEID_SCALEFACTORY = 29,

	/**
	\brief The value represents the minimum packet size.
	*/	
	BGAPI_VALUEID_MINPACKETSIZE = 30,

	/**
	\brief The value represents the maximum packet size.
	*/	
	BGAPI_VALUEID_MAXPACKETSIZE = 31,

	/**
	\brief The value represents a color.
	*/	
	BGAPI_VALUEID_COLOR = 32,

	/**
	\brief The value represents a pixelformat. See BGAPI_PIXTYPE_xxx deifnitions.
	*/	
	BGAPI_VALUEID_PIXELFORMAT = 33,

	/**
	\brief The value represents a number of sth. (e.g. number of buffers)
	*/
	BGAPI_VALUEID_NUMBEROF = 34,

	/**
	\brief the value represents a flag.
	*/
	BGAPI_VALUEID_FLAG = 35,

	/**
	\brief Don't use that.
	*/	
    BGAPI_VALUEID_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_ValueType is used by value to define the type of the actual used value itself.
*/
enum BGAPI_ValueType
{
	/**
	\brief The value represents a single value.
	*/	
	BGAPI_VALUETYPE_SINGLE = 0,

	/**
	\brief The value represents a list.
	*/	
	BGAPI_VALUETYPE_LIST = 1,

	/**
	\brief The value represents an array with maximum 32 entries..
	*/	
	BGAPI_VALUETYPE_ARRAY = 2,

	/**
	\brief The value represents an large array with maximum 4096 entries.
	*/	
	BGAPI_VALUETYPE_LARGEARRAY = 3,

	/**
	\brief The value represents a matrix.
	*/	
	BGAPI_VALUETYPE_MATRIX = 4,

	/**
	\brief Don't use that.
	*/	
	BGAPI_VALUETYPE_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_Type is used by Value to define which data type is actually hold inside a
\brief Value object.
*/
enum BGAPI_Type
{
	/**
	\brief The value is not specified. Dummy value.
	*/	
	BGAPI_TYPE_UNDEF = 0,

	/**
	\brief The value is boolean value.
	*/	
    BGAPI_TYPE_BOOL = 1,

	/**
	\brief The value is an integer value.
	*/	
	BGAPI_TYPE_INT = 2,

	/**
	\brief The value is an float value.
	*/	
	BGAPI_TYPE_FLOAT = 3,

	/**
	\brief The value is a string.
	*/	
    BGAPI_TYPE_STRING = 4,

	/**
	\brief The value is byte value.
	*/
	BGAPI_TYPE_BYTE = 5,

	/**
	\brief The value is pointer.
	*/	
    BGAPI_TYPE_PTR = 6,

	/**
	\brief The value is a handle.
	*/	
    BGAPI_TYPE_HANDLE = 7,

	/**
	\brief The value is feature id.
	*/	
	BGAPI_TYPE_FEATURE		= 8,

	/**
	\brief The value is a value. Used for List, Array and Matrix.
	*/	
	BGAPI_TYPE_VALUE = 9,

	/**
	\brief Don't use that.
	*/	
	BGAPI_TYPE_FORCEINT		= 0xFFFFFFFF
};

/**
\brief The struct BGAPI_ParameterInfo provides information about the range and gets the successor 
\brief and the predecessor of a value.
*/
typedef struct {
	/**
	\brief The size of this struct.
	*/	
	int cbSize;

	/**
	\brief The minimum value.
	*/	
	BGAPI::Value * min;

	/**
	\brief The maximum value.
	*/	
	BGAPI::Value * max;

	/**
	\brief The next value.
	*/	
	BGAPI::Value * next;

	/**
	\brief The previous value.
	*/	
	BGAPI::Value * prev;
} BGAPI_ParameterInfo;

/**
\brief The struct BGAPI_ValueInfo holds all needed information to work with values.
*/
typedef struct {
	/**
	\brief The size of this struct.
	*/
	int cbSize;
	
	/**
	\brief The value id. Refer to enumeration BGAPI_ValueId.
	*/
    BGAPI_ValueId id;
	
	/**
	\brief The type of the value. Refer to enumeration BGAPI_Type.
	*/
    BGAPI_Type type;

	/**
	\brief The value type of the value. Refer to enumeration BGAPI_ValueType.
	*/
    BGAPI_ValueType vtype;

	/**
	\brief A brief value description.
	*/
	char description[256];

	/**
	\brief The unit of the value.
	*/
    char unitdescription[32]; 
	
	/**
	\brief A multiplier.
	*/
    float multiplier;
} BGAPI_ValueInfo;

/**
\brief The BGAPI_Demosaic used by Image object to determine which raw data interpolation should be used
\brief by the demosaic function.
*/
enum BGAPI_Demosaic
{
 	/**
	\brief Disable demosaic option.
	*/	   
	BGAPI_DEMOSAIC_DISABLE = 0,

	/**
	\brief Demosaic algorithm nearest neighbor.
	*/	   
	BGAPI_DEMOSAIC_NEARESTNEIGHBOR = 1,

	/**
	\brief Demosaic algorithm bilinear.
	*/	
	BGAPI_DEMOSAIC_BILINEAR = 2,

	/**
	\brief Demosaic algorithm bilinear high quality.
	*/	
	BGAPI_DEMOSAIC_BILINEARHQ = 3,

	/**
	\brief Simple Demosaic in Cpp code
	\brief Don't use that.
	*/		
	BGAPI_DEMOSAIC_SIMPLECPP = 100,
	/**
	\brief Simple Demosaic in Asm code
	\brief Don't use that.
	*/		
	BGAPI_DEMOSAIC_SIMPLEASM = 101,
	/**
	\brief HQ Demosaic to Mono
	*/		
	BGAPI_DEMOSAIC_TOMONO    = 102,
	/**
	\brief Don't use that.
	*/		
	BGAPI_DEMOSAIC_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_Scale used by Image object to perform a scaling option.
*/
enum BGAPI_Scale
{
	/**
	\brief Disable scale option.
	*/	
	BGAPI_SCALE_DISABLE = 0,

	/**
	\brief Scale option replace.
	*/	
	BGAPI_SCALE_REPLACE = 1,

	/**
	\brief Scale option box.
	*/	
	BGAPI_SCALE_BOX = 2,
	
	/**
	\brief Scale option bilinear.
	*/	
	BGAPI_SCALE_BILIN = 3,
	
	/**
	\brief Scale option gauss.
	*/	
	BGAPI_SCALE_GAUSS = 4,
	
	/**
	\brief hScale option amming.
	*/	
	BGAPI_SCALE_HAMMING = 5,
	
	/**
	\brief Scale option blackman.
	*/	
	BGAPI_SCALE_BLACKMAN = 6,
	
	/**
	\brief Don't use that.
	*/
	BGAPI_SCALE_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_GigeTriggerDeviceMode used by Feature BGAPI_FEATUREID_CAMERA_GIGETRIGGERDEVICEMODE to determine which 
\brief mode of triggerdrvice should be used.
*/
enum BGAPI_GigeTriggerDeviceMode
{
	/**
	\brief Device is in normal mode, devietrigger is unused.
	*/	
	BGAPI_GIGETRIGGERDEVICEMODE_OFF = 0,

	/**
	\brief Device is in master mode, deviet sends swtrigger for each hwtrigger.
	*/	
	BGAPI_GIGETRIGGERDEVICEMODE_MASTER = 1,

	/**
	\brief  Device is in slave mode, device triggers on swtrigger send by masterdevice.
	*/
	BGAPI_GIGETRIGGERDEVICEMODE_SLAVE = 2,

	/**
	\brief Don't use that.
	*/
	BGAPI_GIGETRIGGERDEVICEMODE_FORCEINT	= 0xFFFFFFFF
};
/**
\brief The enumeration BGAPI_Polarity used by Feature BGAPI_FEATUREID_CAMERA_TRIGGERPOLARITY and 
\brief BGAPI_FEATUREID_CAMERA_FLASHPOLARITY to determine which polarity should be used.
*/
enum BGAPI_Polarity
{
 	/**
	\brief Low for state controlled, Falling edge for edge controlled.
	*/
	BGAPI_POLARITY_LOW = 0,

	/**
	\brief Hight for state controlled. Rising edge for edge controled.
	*/   
	BGAPI_POLARITY_HIGH = 1,

	/**
	\brief Don't use that.
	*/
	BGAPI_POLARITY_FORCEINT	= 0xFFFFFFFF
};


/**
\brief The enumeration BGAPI_Activation used by Feature BGAPI_FEATUREID_CAMERA_TRIGGERACTIVATION, BGAPI_FEATUREID_CAMERA_FLASHACTIVATION and BGAPI_FEATUREID_CAMERA_LINEACTIVATION
*/
enum BGAPI_Activation
{
 	/**
	\brief Hight for state controlled. Rising edge for edge controled.
	*/   
	BGAPI_ACTIVATION_STATEHIGH = 0,

 	/**
	\brief Hight for state controlled. Rising edge for edge controled.
	*/   
	BGAPI_ACTIVATION_STATELOW = 1,

	/**
	\brief Low for state controlled, Falling edge for edge controlled.
	*/   
	BGAPI_ACTIVATION_RISINGEDGE = 2,

	/**
	\brief Low for state controlled, Falling edge for edge controlled.
	*/   
	BGAPI_ACTIVATION_FALLINGEDGE = 3,

	/**
	\brief Low for state controlled, Falling edge for edge controlled.
	*/   
	BGAPI_ACTIVATION_ANYGEDGE = 4,

	/**
	\brief Don't use that.
	*/
	BGAPI_ACTIVATION_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_TriggerSource used by Feature BGAPI_FEATUREID_TRIGGER_SOURCE to select several trigger sources. For some hardware plattforms  it is possible to select more than one trigger sources at once.
*/
enum BGAPI_TriggerSource
{
	/**
	\brief The trigger source is software.
	*/
	BGAPI_TRIGGERSOURCE_SOFTWARE	= 0,

	/**
	\brief Hardware source 1.
	*/
	BGAPI_TRIGGERSOURCE_HARDWARE1	= 1,

	/**
	\brief Hardware source 2.
	*/
	BGAPI_TRIGGERSOURCE_HARDWARE2 	= 2,

	/**
	\brief Hardware source 3.
	*/
	BGAPI_TRIGGERSOURCE_HARDWARE3 	= 3,

	/**
	\brief Hardware source 4.
	*/
	BGAPI_TRIGGERSOURCE_HARDWARE4 	= 4,

	/**
	\brief Hardware source 5.
	*/
	BGAPI_TRIGGERSOURCE_HARDWARE5 	= 5,

	/**
	\brief Hardware source 6.
	*/
	BGAPI_TRIGGERSOURCE_HARDWARE6 	= 6,

	/**
	\brief Hardware source 7.
	*/
	BGAPI_TRIGGERSOURCE_HARDWARE7 	= 7,

	/**
	\brief Hardware source 8.
	*/
	BGAPI_TRIGGERSOURCE_HARDWARE8 	= 8,

	/**
	\brief Broadcast command trigger.
	*/
	BGAPI_TRIGGERSOURCE_COMMANDTRIGGER = 10,

	/**
	\brief Enable all available trigger sources.
	*/
	BGAPI_TRIGGERSOURCE_ALL = 100,

	/**
	\brief Disable trigger sources.
	*/
	BGAPI_TRIGGERSOURCE_OFF = 101,

	/**
	\brief Don't use that.
	*/
	BGAPI_TRIGGERSOURCE_FORCEINT 	= 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_Shutter used by Feature BGAPI_FEATUREID_CAMERA_SHUTTER to determine which shutter mode 
\brief should be used.
*/
enum BGAPI_Shutter
{
	/**
	\brief Overlapped shutter.
	*/
	BGAPI_SHUTTER_OVERLAPPED = 0,

	/**
	\brief Sequential shutter.
	*/
	BGAPI_SHUTTER_SEQUENTIAL = 1,

	/**
	\brief Rolling shutter.
	*/
	BGAPI_SHUTTER_ROLLING = 2,

	/**
	\brief Simultan exposure of more than one frames.
	*/
	BGAPI_SHUTTER_SIMULTAN	= 3,

	/**
	\brief Don't use that.
	*/
	BGAPI_SHUTTER_FORCEINT	= 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_DataAccess used by Feature BGAPI_FEATUREID_CAMERA_DATAACCESS to determine which mode is offered 
\brief for camera data access.
 */
enum BGAPI_DataAccess
{
	/**
	\brief Normal access. The most current image is signaled. Older images are lost.
	*/
	BGAPI_DATAACCESS_NORMAL = 0,

	/**
	\brief Queue access.
	*/
	BGAPI_DATAACCESS_QUEUE = 3,

	/**
	\brief Don't use that.
	*/
	BGAPI_DATAACCESS_FORCEINT  = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_DataAccessMode used by Feature BGAPI_FEATUREID_CAMERA_DATAACCESS to determine which mode is offered 
\brief for camera data access.
 */
enum BGAPI_DataAccessMode
{
	/**
	\brief Buffer mode normal intern.
	*/
	BGAPI_DATAACCESSMODE_NORMALINTERN = 10,

	/**
	\brief Buffer mode normal extern.
	*/
	BGAPI_DATAACCESSMODE_NORMALEXTERN = 11,

	/**
	\brief Buffer mode queued intern.
	*/
	BGAPI_DATAACCESSMODE_QUEUEDINTERN = 12,

	/**
	\brief Buffer mode queued extern.
	*/
	BGAPI_DATAACCESSMODE_QUEUEDEXTERN = 13,

	/**
	\brief Don't use that.
	*/
	BGAPI_DATAACCESSMODE_FORCEINT  = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_Readout used by Feature BGAPI_FEATUREID_CAMERA_READOUT to determine which mode is offered 
\brief for camera data access.
*/
enum BGAPI_Readout
{
	/**
	\brief Single readout mode. One sensor tap is used for readout.
	*/
	BGAPI_READOUT_SINGLE = 0,

	/**
	\brief Dual readout mode. Two sensor taps are used for readout.
	*/
	BGAPI_READOUT_DUAL = 1,

	/**
	\brief Quad readout mode. Four sensor taps are used for readout.
	*/
	BGAPI_READOUT_QUAD = 2,

	/**
	\brief Don't use that.
	*/
	BGAPI_READOUT_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_TestPattern used by Feature BGAPI_FEATUREID_CAMERA_TESTPATTERN to determine which mode is offered 
\brief for camera test pattern.
*/
enum BGAPI_TestPattern
{
	/**
	\brief Horizontal pattern (sawtooth orientated).
	*/
    BGAPI_TESTPATTERN_HORI	= 0,

	/**
	\brief Vertical pattern (sawtooth orientated).
	*/
	BGAPI_TESTPATTERN_VERT	= 1,

	/**
	\brief Interface specific pattern
	*/
	BGAPI_TESTPATTERN_VERTLINE = 3,
	/**
	\brief Interface specific pattern
	*/
	BGAPI_TESTPATTERN_HORILINE = 4,
	
	/**
	\brief Interface specific pattern
	*/
	BGAPI_TESTPATTERN_INTERFACE = 5,

	/**
	\brief Don't use that.
	*/
	BGAPI_TESTPATTERN_FORCEINT  = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_LookUpTable defines different types of LUT's.
*/
enum BGAPI_LookUpTable
{
	/**
	\brief Luminance.
	*/
    BGAPI_LOOKUPTABLE_LUMINANCE	= 0,

	/**
	\brief Red.
	*/
	BGAPI_LOOKUPTABLE_RED = 1,

	/**
	\brief Green
	*/
	BGAPI_LOOKUPTABLE_GREEN = 2,

	/**
	\brief Blue
	*/
	BGAPI_LOOKUPTABLE_BLUE = 3,

	/**
	\brief Don't use that.
	*/
	BGAPI_LOOKUPTABLE_FORCEINT  = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_TemperatureSensor defines different types of temperature sensors.
*/
enum BGAPI_TemperatureSensor
{
	/**
	\brief temperature sensor 0.
	*/
    BGAPI_TEMPERATURESENSOR_0 = 0,

	/**
	\brief temperature sensor 1.
	*/
    BGAPI_TEMPERATURESENSOR_1 = 1,

	/**
	\brief temperature sensor 2.
	*/
    BGAPI_TEMPERATURESENSOR_2 = 2,

	/**
	\brief temperature sensor 3.
	*/
    BGAPI_TEMPERATURESENSOR_3 = 3,

	/**
	\brief Don't use that.
	*/
	BGAPI_TEMPERATURESENSOR_FORCEINT  = 0xFFFFFFFF
};

/** 
\brief The BGAPI_FeatureState holds the state of a specified feature.
*/
typedef struct {
	/**
	\brief The size of this struct.
	*/
	int cbSize;

	/**
	\brief The feature is defined but not implemented by the bgapi system libraray (e.g. sys_fxlib, sys_gige).
	*/
	bool bIsImpl;

	/**
	\brief This variable shows the general availability of a feature. 
	\brief It is true, if the camera supports the feature, independently of current camera format. 
	\brief E.g. the camera is a color camera and is in binning mode. 
	\brief The availability of the feature white balance remains on true.
	*/
	bool bIsAvail;

	/**
	\brief This variable shows the current availability of a feature. 
	\brief It is true, if a feature is currently not available. 
	\brief E.g. the camera is a color camera and is in binning mode.
	\brief The current availability of feature white balance stays on true.
	*/
	bool bIsLock;

	/**
	\brief This variable shows the on/off state of a feature. Features without ONOFF switch are always on true (e.g. Exposure).
	*/
	bool bIsEnabled;

//	char		sDescription[256];	/*<  string for description - type eventuell als wide char */
//	char		sUnit[32];			/*<  unit of feature, e.g. "C", "m", "ms", ...           */
}BGAPI_FeatureState;

/**
\brief The enumeration BGAPI_GVSDriverModel is used by Feature BGAPI_FEATUREID_SYSTEM_GVSDRIVERMODEL to select the driver model. 
*/
enum BGAPI_GVSDriverModel
{
	/**
	\brief Socket driver.
	*/
    BGAPI_GVSDRIVERMODEL_SOCKET	= 0,

	/**
	\brief Filter driver.
	*/
	BGAPI_GVSDRIVERMODEL_FILTER	= 1,

	/**
	\brief Turbo driver.
	*/
	BGAPI_GVSDRIVERMODEL_TURBO	= 2,

	/**
	\brief Raw driver.
	*/
	BGAPI_GVSDRIVERMODEL_RAW	= 3,

	/**
	\brief Don't use that.
	*/
	BGAPI_GVSDRIVERMODEL_FORCEINT  = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_SequencerRunMode is used by Feature BGAPI_FEATUREID_CAMERA_BOSEQUENCERRUNMODE in Basic API and
\brief with function setSequencerRunMode in Extended API to select the run mode. 
*/
enum BGAPI_SequencerRunMode
{
	/**
	\brief The sequencer runs only one complete cycle and will started by trigger event.
	*/
	BGAPI_SEQUENCERRUNMODE_ONCEBYTRIGGER		= 1,

	/**
	\brief The sequencer runs only one complete cycle and will started directly after the sequencer is started (without trigger event).
	*/
	BGAPI_SEQUENCERRUNMODE_ONCEFREE			= 2,

	/**
	\brief The sequencer runs continuous and will started by trigger event.
	*/
	BGAPI_SEQUENCERRUNMODE_CYCLEBYTRIGGER	= 3,

	/**
	\brief The sequencer runs continuous and will started directly after the sequencer is started (without trigger event).
	*/
	BGAPI_SEQUENCERRUNMODE_CYCLEFREE		= 4,

	/**
	\brief Don't use that.
	*/
	BGAPI_SEQUENCERRUNMODE_FORCEINT			= 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_UserSet is used by BGAPI_FEATUREID_CAMERA_USERSET_xxx features to work with user settings.
*/
enum BGAPI_UserSet
{
	/**
	\brief This enum represents the factory settings of camera.
	*/	
	BGAPI_USERSET_FACTORYSETTINGS = 0,

	/**
	\brief This enum represents a user set.
	*/	
	BGAPI_USERSET_1 = 1,

	/**
	\brief This enum represents a user set.
	*/	
	BGAPI_USERSET_2 = 2,

	/**
	\brief This enum represents a user set.
	*/	
	BGAPI_USERSET_3 = 3,

	/**
	\brief Don't use that.
	*/
	BGAPI_USERSET_FORCEINT = 0xFFFFFFFF
};

/**
\brief The BGAPIX_CameraImageFormat describes a image format of a camera object. This structure is used by extended API.
*/
typedef struct  {
	/**
	\brief The size of this struct.
	*/
	int cbSize;
	
	/**
	\brief The number/identifier of format.
	*/
	int iFormat;

	/**
	\brief Horizontal size of data block.
	*/
	int iSizeX;

	/**
	\brief Vertical size of data block.
	*/
	int iSizeY;

	/**
	\brief Verbal description of format.
	*/
	char sName[256];

	/**
	\brief Scaling factor for ROI. X coordinates relativ to full frame.
	*/
	float iScaleRoiX;

	/**
	\brief Scaling factor for ROI. Y coordinates relativ to full frame.
	*/
	float iScaleRoiY;
	
	/**
	\brief Offset of format in x direction relativ to full frame.
	*/
	int iOffsetX;

	/**
	\brief Offset of format in y direction relativ to full frame.
	*/
	int iOffsetY;

} BGAPIX_CameraImageFormat;

/**
\brief The BGAPIX_CameraPixelFormat describes the pixel format. This structure is used by extended API.
*/
typedef struct
{
	/**
	\brief The size of this struct.
	*/
	int cbSize;

	/**
	\brief Pixel format. See BGAPI_PIXTYPE_xxx definitions.
	*/
	int iPixelFormat;
	
	/**
	\brief Verbal description of format.
	*/
	char sName[256];
	
	/**
	\brief Native image.
	*/
	bool bColor;

	/**
	\brief Number of bytes per pixel.
	*/
	int iPixelBytes;

	/**
	\brief Number of valid bits per pixel.
	*/
	int iPixelBits;

	/**
	\brief Number of bytes of the current used image format.
	*/
	int iImageByteSize;
	/**
	\brief Minimal available packet size for Format_7 in IIDC mode (note: the minimum value is the step size at the same time).
	*/
	int iMinPacketSize;

	/**
	\brief Maximal available packet size for Format_7 in IIDC mode.
	*/
	int iMaxPacketSize;

} BGAPIX_CameraPixelFormat;

/** 
\brief The enumeration BGAPIX_SensorType holds the information for applied sensor type.
*/
enum BGAPIX_SensorType {    
	/**
	\brief Undefined.
	*/
	BGAPIX_SENSORTYPE_UNDEF = 0,

	/**
	\brief CCD Matrix sensor.
	*/
	BGAPIX_SENSORTYPE_MATRIX_CCD = 1,

	/**
	\brief CCD Line sensor.
	*/
	BGAPIX_SENSORTYPE_LINE_CCD = 2,
	
	/**
	\brief CCD Matrix sensor.
	*/
	BGAPIX_SENSORTYPE_MATRIX_CMOS = 3,

	/**
	\brief CCD Line sensor.
	*/
	BGAPIX_SENSORTYPE_LINE_CMOS = 4,

	/**
	\brief Don't use that.
	*/
	BGAPIX_SENSORTYPE_FORCEINT = 0xFFFFFFFF
};

/** 
\brief The enumeration BGAPIX_SensorData holds the information of the raw data format of applied sensor.
*/
enum BGAPIX_SensorData {    
	/**
	\brief Undefined.
	*/
	BGAPIX_SENSORDATA_UNDEF = 0,

	/**
	\brief Monochrome sensor.
	*/
	BGAPIX_SENSORDATA_MONO = 1,

	/**
	\brief Color bayer.
	*/
	BGAPIX_SENSORDATA_COLOR = 2,

	/**
	\brief Don't use that.
	*/
	BGAPIX_SENSORDATA_FORCEINT = 0xFFFFFFFF
};

/** 
\brief The enumeration BGAPIX_DeviceType holds the information for applied device type.
*/
enum BGAPIX_DeviceType {
	/**
	\brief Undefined.
	*/
    BGAPIX_DEVICETYPE_UNDEF = 0,

	/**
	\brief Baumer PCI card.
	*/
	BGAPIX_DEVICETYPE_BAUMERPCI = 1,

	/**
	\brief Common PCI card, OHCI controller
	*/
	BGAPIX_DEVICETYPE_COMMONPCI_OHCI = 2,
	
	/**
	\brief Common PCI card, NIC
	*/
	BGAPIX_DEVICETYPE_COMMONPCI_NIC = 3,

	/**
	\brief CameraLink PCI card 
	*/
	BGAPIX_DEVICETYPE_COMMONPCI_CAMERALINK = 4,

	/**
	\brief Baumer camera based on Baumer Link.
	*/
	BGAPIX_DEVICETYPE_BAUMERCAM_IX = 10,

	/**
	\brief Baumer FireWire camera based on FCAM, FCAM2.
	*/
	BGAPIX_DEVICETYPE_BAUMERCAM_FCAMX = 11,

	/**
	\brief Baumer FireWire camera based on DCAM standard.
	*/
	BGAPIX_DEVICETYPE_BAUMERCAM_DCAM = 12,

	/**
	\brief Baumer GigaBit Ethernet camera.
	*/
	BGAPIX_DEVICETYPE_BAUMERCAM_GIGE = 13,

	/**
	\brief Baumer CameraLink camera. 
	*/
	BGAPIX_DEVICETYPE_BAUMERCAM_CAMERALINK = 14,

	/**
	\brief Common CameraLink camera. 
	*/
	BGAPIX_DEVICETYPE_COMMONCAM_CAMERALINK = 15,

	/**
	\brief Common camera.
	*/
	BGAPIX_DEVICETYPE_COMMONCAM = 16,

	/**
	\brief Don't use that.
	*/
	BGAPIX_DEVICETYPE_FORCEINT = 0xFFFFFFFF
};

/** 
\brief The enumeration BGAPI_ImageStatus get status of image transmission.
*/
enum BGAPI_ImageStatus {
	/**
	\brief Image ok.
	*/
    BGAPI_IMAGESTATUS_GOOD = 0,

	/**
	\brief Timeout occured.
	*/
	BGAPI_IMAGESTATUS_TIMEOUT = 1,

	/**
	\brief Missing packets.
	*/
	BGAPI_IMAGESTATUS_MISSINGPACKETS = 2,
	
	/**
	\brief Invalid header.
	*/
	BGAPI_IMAGESTATUS_INVALIDHEADER = 3,

	/**
	\brief Invalid image settings.
	*/
	BGAPI_IMAGESTATUS_INVALIDSETTINGS = 4,

	/**
	\brief Invalid image settings.
	*/
	BGAPI_IMAGESTATUS_NOTTRANSFORMED = 5,

	/**
	\brief Don't use that.
	*/
	BGAPI_IMAGESTATUS_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_FlipType used by Feature BGAPI_FEATUREID_CAMERA_FLIPX and 
\brief BGAPI_FEATUREID_CAMERA_FLIPY to defines the possible BGAPI image flipping modes.
*/
enum BGAPI_FlipType
{
 	/**
	\brief Image is not flipped.
	*/
	BGAPI_FLIPTYPE_NONE = 0,

	/**
	\brief Image is horizontal flipped.
	*/   
	BGAPI_FLIPTYPE_X = 1,

	/**
	\brief Image is vertical flipped.
	*/   
	BGAPI_FLIPTYPE_Y = 2,

	/**
	\brief Image is horizontal and vertical flipped.
	*/   
	BGAPI_FLIPTYPE_XY = 3,

	/**
	\brief Don't use that.
	*/
	BGAPI_FLIPTYPE_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_ToFModulationFrequency defines several modulation frequencies of ToF operation.
*/
enum BGAPI_ToFModulationFrequency
{
 	/**
	\brief 14.5 Mhz.
	*/
	BGAPI_TOFMODULATIONFREQUENCY_14_5 = 0,
 	/**
	\brief 15 Mhz.
	*/
	BGAPI_TOFMODULATIONFREQUENCY_15 = 1,
 	/**
	\brief 15.5 Mhz.
	*/
	BGAPI_TOFMODULATIONFREQUENCY_15_5 = 2,
 	/**
	\brief 29 Mhz.
	*/
	BGAPI_TOFMODULATIONFREQUENCY_29 = 3,
 	/**
	\brief 30 Mhz.
	*/
	BGAPI_TOFMODULATIONFREQUENCY_30 = 4,
 	/**
	\brief 31 Mhz.
	*/
	BGAPI_TOFMODULATIONFREQUENCY_31 = 5,
	/**
	\brief Don't use that.
	*/
	BGAPI_TOFMODULATIONFREQUENCY_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_Filter used by Camera object (e.g. TZG) to perform a camera filter options.
*/
enum BGAPI_Filter
{
	/**
	\brief Phase fixed pattern compensation.
	*/	
	BGAPI_FILTER_PHASEFIEXEDPATTERN = 0,

	/**
	\brief Phase non-linearity compensation.
	*/	
	BGAPI_FILTER_PHASENONLINEARITY = 1,

	/**
	\brief Temperature compensation.
	*/	
	BGAPI_FILTER_TEMPERATURECOMPENSATION = 2,

	/**
	\brief Internal time compensation.
	*/	
	BGAPI_FILTER_INTERNALTIME = 3,

	/**
	\brief Optical feedback line compensation.
	*/	
	BGAPI_FILTER_OPTICALFEEDBACKLINE = 4,
	
	/**
	\brief Adaptive neighborhood filter.
	*/	
	BGAPI_FILTER_ADAPTIVENEIGHBORHOOD = 5,

	/**
	\brief Defect pixel filter.
	*/	
	BGAPI_FILTER_DEFECTPIXEL = 6,

	/**
	\brief Don't use that.
	*/
	BGAPI_FILTER_FORCEINT = 0xFFFFFFFF
};


/** 
\brief The BGAPIX_TypeBOOL holds information of a simple "bool" feature.
*/
typedef struct {  
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;

	/**
	\brief Boolean value.
	*/
	bool bcur;
}BGAPIX_TypeBOOL;

/** 
\brief The BGAPIX_TypeINT holds information of a sinple "integer" feature,
\brief beside the current value 'current' masks for allowed read and write operation
\brief are available in 'maskread' and 'maskwrite'.
\brief E.g. if 'maskwrite == 0x00000000' and 'maskwrite == 0x0000FFFF' 
\brief no write operation to this feature is possible, but the lower 16 bits are readable 
*/
typedef struct {  
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Current integer value.
	*/
	int current;

	/**
	\brief Read mask.
	*/
	int maskread;

	/**
	\brief Write mask.
	*/
	int maskwrite;
} BGAPIX_TypeINT;

/** 
\brief The BGAPIX_TypeRangeINT holds information of an "scalar integer" feature.
\brief The range of value is characterised by 'minimum' and 'maximum', 'next' and 'prev'
\brief are the incremental or decremental values to reach the next ot previous value.
*/
typedef struct {			   
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Current range value.
	*/
	int current;

	/**
	\brief Defines the minimum of value.
	*/
	int minimum;

	/**
	\brief Defines the maximum of value.
	*/
	int maximum;

	/**
	\brief Defines the next possible value.
	*/
	int next;

	/**
	\brief Defines the previous possible value.
	*/
	int prev;
} BGAPIX_TypeRangeINT;

/** 
\brief The BGAPIX_TypeRangeFLOAT holds information of an "scalar float" feature.
\brief The range of value is characterised by 'minimum' and 'maximum', 'next' and 'prev'
\brief are the incremental or decremental values to reach the next ot previous value.
*/
typedef struct {			   
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Current range value.
	*/
	float current;

	/**
	\brief Defines the minimum of value.
	*/
	float minimum;

	/**
	\brief Defines the maximum of value.
	*/
	float maximum;

	/**
	\brief Defines the next possible value.
	*/
	float next;

	/**
	\brief Defines the previous possible value.
	*/
	float prev;
} BGAPIX_TypeRangeFLOAT;


/** 
\brief The BGAPIX_TypeSTRING holds information of a single byte char string.
*/
typedef struct {  
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief String value.
	*/
	char * current;

	/**
	\brief string length.
	*/
	int length;
} BGAPIX_TypeSTRING;

/** 
\brief The BGAPIX_TypeROI holds information of an "region of interest " feature.
\brief The range of value is characterised by values 
\brief 'mintop', 'minleft', 'minextx', 'minexty' and 
\brief 'maxtop', 'maxleft', 'maxextx', 'maxexty'
*/
typedef struct {  
	/**
	\brief Size of this struct.
	*/
	int					cbSize;			/*< size of this struct		*/

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId		fident;			/*< identifier of feature   */
		
	/**
	\brief Current top value.
	*/
	int curtop;

	/**
	\brief Current left value.
	*/
	int curleft;

	/**
	\brief Current right value.
	*/
	int curright;

	/**
	\brief Current bottom value.
	*/
	int curbottom;

	/**
	\brief Minimum of top value.
	*/
	int mintop;

	/**
	\brief Minimum of left value.
	*/
	int minleft;

	/**
	\brief Minimum of right value.
	*/
	int minright;

	/**
	\brief Minimum of bottom value.
	*/
	int minbottom;

	/**
	\brief Minimum of top value.
	*/
	int maxtop;

	/**
	\brief Minimum of left value.
	*/
	int maxleft;

	/**
	\brief Minimum of right value.
	*/
	int maxright;

	/**
	\brief Minimum of bottom value.
	*/
	int maxbottom;
} BGAPIX_TypeROI;

/** 
\brief The BGAPIX_TypeArrayINT holds information of an "int array" feature.
\brief The size of array is fix and limited to 32 entrys.
\brief An indicator of used values is represented by 'length' 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages
*/
typedef struct {			
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Used size of array.
	*/
	int length;

	/**
	\brief Array of type integer with maximum 32 values.
	*/
	int array[32];
} BGAPIX_TypeArrayINT;

/**
\brief The BGAPIX_TypeArrayFLOAT holds information of an "float array " feature.
\brief The size of array is fix and limited to 32 entrys.
\brief An indicator of used values is represented by 'length' 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages
*/
typedef struct {			
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Used size of array.
	*/
	int length;

	/**
	\brief Array of type float with maximum 32 values.
	*/
	float array[32];
} BGAPIX_TypeArrayFLOAT;

/** 
\brief The BGAPIX_TypeArrayBYTE holds information of an "byte array" feature.
\brief The size of array is fix and limited to 32 entrys.
\brief An indicator of used values is represented by 'length' 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages
*/
typedef struct {			
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Used size of array.
	*/
	int length;

	/**
	\brief Array with maximum 32 values.
	*/
	unsigned char array[32];
} BGAPIX_TypeArrayBYTE;

/** 
\brief The BGAPIX_TypeArrayBOOL holds information of an "bool array" feature.
\brief The size of array is fix and limited to 32 entrys.
\brief An indicator of used values is represented by 'length' 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages
*/
typedef struct {			
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Used size of array.
	*/
	int length;

	/**
	\brief Array with maximum 32 values.
	*/
	bool array[32];
} BGAPIX_TypeArrayBOOL;

/** 
\brief The BGAPIX_TypeLargeArrayINT holds information of an "int array " feature.
\brief The size of array is fix and limited to 4096 entrys.
\brief An indicator of used values is represented by 'length' 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages
*/
typedef struct {			
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Used size of array.
	*/
	int length;

	/**
	\brief Array of type integer with maximum 4096 values.
	*/
	int array[4096];
} BGAPIX_TypeLargeArrayINT;

/** 
\brief The BGAPIX_TypeLargeArrayFLOAT holds information of an "float array " feature.
\brief The size of array is fix and limited to 4096 entrys.
\brief An indicator of used values is represented by 'length' 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages
*/
typedef struct {			
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Used size of array.
	*/
	int length;

	/**
	\brief Array of type float with maximum 4096 values.
	*/
	float array[4096];
} BGAPIX_TypeLargeArrayFLOAT;

/** 
\brief The BGAPIX_TypeLargeArrayBOOL holds information of an "bool array " feature.
\brief The size of array is fix and limited to 4096 entrys.
\brief An indicator of used values is represented by 'length' 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages
*/
typedef struct {			
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Used size of array.
	*/
	int length;

	/**
	\brief Array of type float with maximum 4096 values.
	*/
	bool array[4096];
} BGAPIX_TypeLargeArrayBOOL;

/** 
\brief The BGAPIX_TypeLargeArrayBYTE holds information of an "byte array " feature.
\brief The size of array is fix and limited to 4096 entrys.
\brief An indicator of used values is represented by 'length' 
\brief The reason of a static size is the portability of structures and fuctions to 
\brief other programming languages
*/
typedef struct {			
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Used size of array.
	*/
	int length;

	/**
	\brief Array with maximum 4096 values.
	*/
	unsigned char array[4096];
} BGAPIX_TypeLargeArrayBYTE;

/** 
\brief The BGAPIX_TypeMatrixINT holds information in a matrix of int values.
\brief The size of matrix is fix and limited to 1024 (e.g. 32*32) entrys.
\brief An indicator of used dimensions for x and y directions is represented by 'xdim' and 'ydim'. 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages.
*/
typedef struct {
	/**
	\brief Size of this struct.
	*/	
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;

	/**
	\brief Matrix dimension in x direction.
	*/
	int xdim;

	/**
	\brief Matrix dimension in y direction.
	*/
	int ydim;
	/**
	\brief The matrix itself. Maximum dimension 32*32. 
	*/
	int matrix[1024];
} BGAPIX_TypeMatrixINT;

/** 
\brief The BGAPIX_TypeMatrixFLOAT holds information in a matrix of float values.
\brief The size of matrix is fix and limited to 1024 (e.g. 32*32) entrys.
\brief An indicator of used dimensions for x and y directions is represented by 'xdim' and 'ydim'. 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages.
*/
typedef struct {
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;

	/**
	\brief Matrix dimension in x direction.
	*/
	int xdim;

	/**
	\brief Matrix dimension in y direction.
	*/
	int ydim;

	/**
	\brief The matrix itself. Maximum dimension 32*32. 
	*/
	float matrix[1024];
} BGAPIX_TypeMatrixFLOAT;

/** 
\brief The BGAPIX_TypeListFEATURE holds information of an "feature list" for subfeatures.
\brief The size of array is fix and limited to 32 entrys.
\brief An indicator of used values is represented by 'length' 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages
*/
typedef struct {			
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Current feature index.
	*/
	int current;
	
	/**
	\brief Used size of enumeration array.
	*/
	int length;

	/**
	\brief Enumeration array.
	*/
	BGAPI_FeatureId array[32];
} BGAPIX_TypeListFEATURE;

/** 
\brief The BGAPIX_TypeListINT holds information of an "list" for values.
\brief The size of array is fix and limited to 32 entrys.
\brief An indicator of used values is represented by 'length' 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages
*/
typedef struct {			
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Current list index.
	*/
	int current;

	/**
	\brief Used size of list.
	*/
	int length;

	/**
	\brief List of integer values. Maximum is 32 entries.
	*/
	int array[32];
} BGAPIX_TypeListINT;

/** 
\brief The BGAPIX_TypeListFLOAT holds information of an "list" for values.
\brief The size of array is fix and limited to 32 entrys.
\brief An indicator of used values is represented by 'length' 
\brief The reason of a static size is the portability of structures and functions to 
\brief other programming languages
*/
typedef struct {			
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Identifier of feature.
	*/
	BGAPI_FeatureId fident;
	
	/**
	\brief Current list index.
	*/
	int current;

	/**
	\brief Used size of list.
	*/
	int length;

	/**
	\brief List of float values. Maximum is 32 entries.
	*/
	float array[32];
} BGAPIX_TypeListFLOAT;

/** 
\brief Global info for physical devices  
*/
typedef struct  {
	/**
	\brief Size of this struct.
	*/
	int cbSize;
		
	/**
	\brief Camera, PCI-interface (see BGAPIX_DeviceType).
	*/
	BGAPIX_DeviceType type;

	/**
	\brief Verbal device description. Maximum length is 64 bytes.
	*/
	char vendorName[64];

	/**
	\brief E.g. Baumer ieee id, ...
	*/
	int vendor;

	/**
	\brief Additional manufactor informations. Maximum length is 64 bytes.
	*/
	char manufactorInfo[64];

	/**
	\brief Verbal device description. Maximum length is 64 bytes.
	*/
	char modelName[64];

	/**
	\brief Lower 16 Bit used for id of device.
	\brief Upper 16 Bit used for sub id of device.
	*/
	int modelIdent;

	/**
	\brief Serial number. Managed by HW. First entry is the lower part of serial number.
	*/
	char serialNumber[64];

	/**
	\brief User defined camera label. Maximum length is 64 bytes.
	*/
	char label[64];

	/**
	\brief Color/monochrom camera (see BGAPIX_SensorData).
	*/
	BGAPIX_SensorData eSensorData;

	/**
	\brief Matrix/line camera (see BGAPIX_SensorType).
	*/
	BGAPIX_SensorType eSensorType;
	
	/**
	\brief Specification number. The upper word is major number and the lower word is minor number.
	*/
	int specVersion;

	/**
	\brief Firmware version as string. Maximum 64 bytes.
	*/
	char firmwareVersion[64];

	/**
	\brief Hardware version as string. Maximum is 64 bytes.
	*/
	char hardwareVersion[64];
	
	/**
	\brief IP Adresse of device.
	*/
	unsigned char IP_Address[4];

	/**
	\brief IP Subnetmask of device.
	*/
	unsigned char IP_SubnetMask[4];

	/**
	\brief IP Gateway address.
	*/
	unsigned char IP_Gateway[4];

	/**
	\brief MAC Adresse of camera.
	*/
	unsigned char MAC_Address[6];

	/**
	\brief Connected port of higher ranking interface.
	*/
	int port;

	/**
	\brief Payloadsize of camera device.
	*/
	int payloadsize;
	
	/**
	\brief Logical host number (OHCI controller) with the camera is connected, for camera types BGAPIX_DEVICETYPE_BAUMERCAM_FCAMX and BGAPIX_DEVICETYPE_BAUMERCAM_DCAM.
	\brief Logical host number (NIC controller) with the camera is connected, for camera type BGAPIX_DEVICETYPE_BAUMERCAM_GIGE.
	*/
	int hostnr;

	/**
	\brief Maximum transmission unit. Only supported by BGAPIX_DEVICETYPE_BAUMERCAM_GIGE.
	*/
	int mtu;

	/**
	\brief Don't use that.
	*/
	int reserved[16];
} BGAPIX_CameraInfo;

/** 
\brief Global info for physical devices  
*/
typedef struct  {
	/**
	\brief Size of this struct.
	*/
	int cbSize;
		
	/**
	\brief Camera, PCI-interface (see BGAPIX_DeviceType).
	*/
	BGAPIX_DeviceType type;

	/**
	\brief Verbal device description. Maximum length is 64 bytes.
	*/
	char vendorName[64];
	
	/**
	\brief E.g. Baumer ieee id, ...
	*/
	int vendor;

	/**
	\brief Serial number. Managed by HW. First entry is the lower part of serial number.
	*/
	char serialNumber[64];

	/**
	\brief Version string. Shows the versions of used system module and drivers.
	*/
	char versionStr[256];

	/**
	\brief Current used payload size for send.
	*/
	int payload_send;

	/**
	\brief Current used payload size for receive.
	*/
	int payload_recv;

}BGAPIX_SystemInfo;

/** 
\brief Global camera statistic  
*/
typedef struct  {
	/**
	\brief Size of this struct.
	*/
	int cbSize;
		
	/**
	\brief Camera type (see BGAPIX_DeviceType).
	*/
	BGAPIX_DeviceType type;

	/**
	\brief Definition of statistic array for type = BGAPIX_DEVICETYPE_BAUMERCAM_GIGE
	\brief statistic[0] Received Frames Good
	\brief statistic[1] Received Frames Corrupted
	\brief statistic[2] Lost Frames
	\brief statistic[3] Resend Requests
	\brief statistic[4] Resend Packets
	\brief statistic[5] Lost Packest
	\brief statistic[6] Bandwidth
	\brief statistic[7-31] Reserved
	\brief Definition of statistic array for type = BGAPIX_DEVICETYPE_BAUMERCAM_FCAMX or BGAPIX_DEVICETYPE_BAUMERCAM_DCAM
	\brief statistic[0-31] Reserved
	*/
	int statistic[32];
}BGAPIX_CameraStatistic;

/** 
\brief Image information
*/
typedef struct  {
	/**
	\brief Size of this struct.
	*/
	int cbSize;

	/**
	\brief Definition of image status
	\brief status[0] Resend Requests
	\brief status[1] Received Resend Packets 
	\brief status[2] Lost Packets
	\brief status[3] Total Packets
	\brief statistic[4-31] Reserved
	*/
	int information[32];
}BGAPI_ImageInformation;

/*	Pixel Type Defines
	---------------------------------------
	see GigE Vision Specification Draft 1.5.0 January 2, 2006 chapter 26 of GigE Vision Standard

	The following provides defines for the various pixel types supported by bgapi. Each pixel type is
	represented by a 32-bit value. The upper 8-bit indicates the color. The second upper 8-bit indicates the
	number of bit occupied by a pixel (including any padding). This can be used to quickly compute the amount
	of memory required to store an image using this pixel type.

	This definitions are valid for bgapi iamge transfers !!! 
*/

//===================================================
// PIXEL TYPES
//===================================================

/**
\brief Indicate if the pixel is monochrome.
*/
#define BGAPI_PIXTYPE_MONO						0x01000000
/**
\brief Indicate if pixel is RGB.
*/
#define BGAPI_PIXTYPE_COL						0x02000000
/**
\brief Custom, if BGAPI_PIXTYPE_CUSTOM is combined with BGAPI_PIXTYPE_COL see PIXEL_ARANGE.
*/
#define BGAPI_PIXTYPE_CUSTOM						0x80000000
/**
\brief Color mask.
*/
#define BGAPI_PIXTYPE_COLOR_MASK					0x7F000000

// Indicate effective number of bits occupied by the pixel (including padding).
// This can be used to compute amount of memory required to store an image.
/**
\brief 8 bits are occupied by the pixel (including padding).
*/
#define BGAPI_PIXTYPE_OCCUPY8BIT					0x00080000
/**
\brief 10 bits are occupied by the pixel (including padding).
*/
#define BGAPI_PIXTYPE_OCCUPY10BIT				0x000A0000
/**
\brief 12 bits are occupied by the pixel (including padding).
*/
#define BGAPI_PIXTYPE_OCCUPY12BIT				0x000C0000
/**
\brief 16 bits are occupied by the pixel (including padding).
*/
#define BGAPI_PIXTYPE_OCCUPY16BIT				0x00100000
/**
\brief 20 bits are occupied by the pixel (including padding).
*/
#define BGAPI_PIXTYPE_OCCUPY20BIT				0x00140000
/**
\brief 24 bits are occupied by the pixel (including padding).
*/
#define BGAPI_PIXTYPE_OCCUPY24BIT				0x00180000
/**
\brief 32 bits are occupied by the pixel (including padding).
*/
#define BGAPI_PIXTYPE_OCCUPY32BIT				0x00200000
/**
\brief 36 bits are occupied by the pixel (including padding).
*/
#define BGAPI_PIXTYPE_OCCUPY36BIT				0x00240000
/**
\brief 48 bits are occupied by the pixel (including padding).
*/
#define BGAPI_PIXTYPE_OCCUPY48BIT				0x00300000
/**
\brief Pixel size mask.
*/
#define BGAPI_PIXTYPE_EFFECTIVE_PIXEL_SIZE_MASK	0x00FF0000
/**
\brief Pixel size mask shifting.
*/
#define BGAPI_PIXTYPE_EFFECTIVE_PIXEL_SIZE_SHIFT 16

/**
\brief Pixel ID: lower 16-bit of the pixel type.
*/
#define BGAPI_PIXTYPE_ID_MASK					0x0000FFFF

// Mono buffer format defines
// -----------------------------
/**
\brief Mono 8.
*/
#define BGAPI_PIXTYPE_MONO8			(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY8BIT  | 0x0001)

/**
\brief Mono 8 signed.
*/
#define BGAPI_PIXTYPE_MONO8_SIGNED	(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY8BIT  | 0x0002)
/**
\brief Mono 10.
*/
#define BGAPI_PIXTYPE_MONO10			(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0003)
/**
\brief Mono 10 packed.
*/
#define BGAPI_PIXTYPE_MONO10_PACKED	(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY12BIT | 0x0004)
/**
\brief Mono 12.
*/
#define BGAPI_PIXTYPE_MONO12			(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0005)
/**
\brief Mono 12 packed.
*/
#define BGAPI_PIXTYPE_MONO12_PACKED	(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY12BIT | 0x0006)
/**
\brief Mono 14.
*/
#define BGAPI_PIXTYPE_MONO14			(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0025)
/**
\brief Mono 16.
*/
#define BGAPI_PIXTYPE_MONO16			(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0007)

//  Bayer buffer format defines
// ----------------------------
/**
\brief Bayer green red 8.
*/
#define BGAPI_PIXTYPE_BAYGR8			(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY8BIT  | 0x0008)
/**
\brief Bayer red green 8.
*/
#define BGAPI_PIXTYPE_BAYRG8			(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY8BIT  | 0x0009)
/**
\brief Bayer green blue 8.
*/
#define BGAPI_PIXTYPE_BAYGB8			(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY8BIT  | 0x000A)
/**
\brief Bayer blue green 8.
*/
#define BGAPI_PIXTYPE_BAYBG8			(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY8BIT  | 0x000B)

/**
\brief Bayer green red 10.
*/
#define BGAPI_PIXTYPE_BAYGR10		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x000C)
/**
\brief Bayer red green 10.
*/
#define BGAPI_PIXTYPE_BAYRG10		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x000D)
/**
\brief Bayer green blue 10.
*/
#define BGAPI_PIXTYPE_BAYGB10		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x000E)
/**
\brief Bayer blue green 10.
*/
#define BGAPI_PIXTYPE_BAYBG10		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x000F)

/**
\brief Bayer green red 12.
*/
#define BGAPI_PIXTYPE_BAYGR12		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0010)
/**
\brief Bayer red green 12.
*/
#define BGAPI_PIXTYPE_BAYRG12		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0011)
/**
\brief Bayer green blue 12.
*/
#define BGAPI_PIXTYPE_BAYGB12		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0012)
/**
\brief Bayer blue green 12.
*/
#define BGAPI_PIXTYPE_BAYBG12		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0013)

/**
\brief Bayer blue green 10 packed.
*/
#define BGAPI_PIXTYPE_BAYBG10_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY12BIT | 0x0026)
/**
\brief Bayer green blue 10 packed.
*/
#define BGAPI_PIXTYPE_BAYGB10_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY12BIT | 0x0027)
/**
\brief Bayer green red 10 packed.
*/
#define BGAPI_PIXTYPE_BAYGR10_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY12BIT | 0x0028)
/**
\brief Bayer red green 10 packed.
*/
#define BGAPI_PIXTYPE_BAYRG10_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY12BIT | 0x0029)
/**
\brief Bayer blue green 12 packed.
*/
#define BGAPI_PIXTYPE_BAYBG12_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY12BIT | 0x002a)
/**
\brief Bayer green blue 12 packed.
*/
#define BGAPI_PIXTYPE_BAYGB12_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY12BIT | 0x002b)
/**
\brief Bayer green red 12 packed.
*/
#define BGAPI_PIXTYPE_BAYGR12_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY12BIT | 0x002c)
/**
\brief Bayer red green 12 packed.
*/
#define BGAPI_PIXTYPE_BAYRG12_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY12BIT | 0x002d)
/**
\brief Bayer green red 16 packed.
*/
#define BGAPI_PIXTYPE_BAYGR16_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x002e)
/**
\brief Bayer red green 16 packed.
*/
#define BGAPI_PIXTYPE_BAYRG16_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x002f)
/**
\brief Bayer green blue 16 packed.
*/
#define BGAPI_PIXTYPE_BAYGB16_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0030)
/**
\brief Bayer blue green 16 packed.
*/
#define BGAPI_PIXTYPE_BAYBG16_PACKED		(BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0031)

// RGB Packed buffer format defines
// --------------------------------
/**
\brief RGB 8 bit packed.
*/
#define BGAPI_PIXTYPE_RGB8_PACKED	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY24BIT | 0x0014)
/**
\brief BGR 8 bit packed.
*/
#define BGAPI_PIXTYPE_BGR8_PACKED	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY24BIT | 0x0015)
/**
\brief RGBA 8 bit packed.
*/
#define BGAPI_PIXTYPE_RGBA8_PACKED	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY32BIT | 0x0016)
/**
\brief BGRA 8 bit packed.
*/
#define BGAPI_PIXTYPE_BGRA8_PACKED	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY32BIT | 0x0017)
/**
\brief RGB 10 bit packed.
*/
#define BGAPI_PIXTYPE_RGB10_PACKED	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY48BIT | 0x0018)
/**
\brief BGR 10 bit packed.
*/
#define BGAPI_PIXTYPE_BGR10_PACKED	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY48BIT | 0x0019)
/**
\brief RGB 12 bit packed.
*/
#define BGAPI_PIXTYPE_RGB12_PACKED	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY48BIT | 0x001A)
/**
\brief BGR 12 bit packed.
*/
#define BGAPI_PIXTYPE_BGR12_PACKED	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY48BIT | 0x001B)
/**
\brief RGB 16 bit packed.
*/
#define BGAPI_PIXTYPE_RGB16_PACKED	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY48BIT | 0x0033)
/**
\brief RGB 10 bit V1 packed.
*/
#define BGAPI_PIXTYPE_RGB10V1_PACKED (BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY32BIT | 0x001C)
/**
\brief RGB 10 bit V2 packed.
*/
#define BGAPI_PIXTYPE_RGB10V2_PACKED (BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY32BIT | 0x001D)
/**
\brief RGB 12 bit V1 packed.
*/
#define BGAPI_PIXTYPE_RGB12V1_PACKED (BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY36BIT | 0x0034)

// YUV Packed buffer format defines
// --------------------------------
/**
\brief YUV 4:1:1 packed.
*/
#define BGAPI_PIXTYPE_YUV411_PACKED  (BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY12BIT | 0x001E)
/**
\brief YUV 4:2:2 packed.
*/
#define BGAPI_PIXTYPE_YUV422_PACKED  (BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY16BIT | 0x001F)
/**
\brief YUV 4:2:2 YUYV packed.
*/
#define BGAPI_PIXTYPE_YUV422_YUYV_PACKED  (BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0032)
/**
\brief YUV 4:4:4 packed.
*/
#define BGAPI_PIXTYPE_YUV444_PACKED  (BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY24BIT | 0x0020)

// RGB Planar buffer format defines
// --------------------------------
/**
\brief RGB 8 bit planar.
*/
#define BGAPI_PIXTYPE_RGB8_PLANAR	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY24BIT | 0x0021)
/**
\brief RGB 10 bit planar.
*/
#define BGAPI_PIXTYPE_RGB10_PLANAR	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY48BIT | 0x0022)
/**
\brief RGB 12 bit planar.
*/
#define BGAPI_PIXTYPE_RGB12_PLANAR	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY48BIT | 0x0023)
/**
\brief RGB 16 bit V2 planar.
*/
#define BGAPI_PIXTYPE_RGB16_PLANAR	(BGAPI_PIXTYPE_COL  | BGAPI_PIXTYPE_OCCUPY48BIT | 0x0024)

//===================================================
// PIXEL_ARANGE
//===================================================

// if rawdata pixel are transmitted, they are not always in a well defined order.
// under standards IIDC or GigE Vision Standard it is not the case, but under different
// proprietare models we can find such data arangements, especially if pure bayer datas are 
// transmitted from the sensor

// if the BGAPI_PIXTYPE_CUSTOM bit is set in combination with in the upper definitions
// an additional information for the pixel arrangement is neccessary

/**
\brief Mono 8 bit custom format with LUT.
*/
#define BGAPI_PIXTYPE_MONO8_LUT			(BGAPI_PIXTYPE_CUSTOM | BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY8BIT  | 0x0001)

/**
\brief Mono 10 bit custom format with LUT.
*/
#define BGAPI_PIXTYPE_MONO10_LUT			(BGAPI_PIXTYPE_CUSTOM | BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0003)

/**
\brief Mono 12 bit custom format with LUT.
*/
#define BGAPI_PIXTYPE_MONO12_LUT			(BGAPI_PIXTYPE_CUSTOM | BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0005)

/**
\brief Mono 12 bit custom format with LUT.
*/
#define BGAPI_PIXTYPE_MONO16_LUT			(BGAPI_PIXTYPE_CUSTOM | BGAPI_PIXTYPE_MONO | BGAPI_PIXTYPE_OCCUPY16BIT | 0x0007)

/**
\brief Frame mask for number of frames.
*/

#define BGAPI_PIXARANGE_NUMBERMASK				0x0000000F
/**
\brief Order mask for type of arrangement.
*/
#define BGAPI_PIXARANGE_ORDERMASK				0x0000FF0F
/**
\brief Complete bayer field as 2x2.
*/
#define BGAPI_PIXARANGE_BAYER2X2					0x00000100
/**
\brief Complete bayer field as 2x2 in alternate frame sequences. Start pattern is BGAPI_PIX_BAY[GR,RG,GB,BG][8,10,12].
*/
#define BGAPI_PIXARANGE_BAYER2X2ALT				0x00000200
/**
\brief Complete bayer field as 1x4.
*/
#define BGAPI_PIXARANGE_BAYER1X4					0x00000020

/**
\brief The enumeration BGAPI_TraceTarget defines several kinds to control trace output.
*/
enum BGAPI_TraceTarget
{
	/**
	\brief Trace output to file.
	*/
	BGAPI_TRACETARGET_FILE = 0,

	/**
	\brief Trace output to debugger.
	*/
	BGAPI_TRACETARGET_DEBUGGER = 1,

	/**
	\brief Trace output via callback.
	*/
	BGAPI_TRACETARGET_CALLBACK = 2,

	/**
	\brief Don't use that.
	*/
	BGAPI_TRACETARGET_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_TraceSupport defines several kinds to control trace output.
*/
enum BGAPI_TraceSupport
{
	/**
	\brief All trace settings are the same for all available trace objects.
	*/
	BGAPI_TRACESUPPORT_GLOBAL = 0,
	/**
	\brief All trace settings for object of the one class are the same.
	*/
	BGAPI_TRACESUPPORT_CLASS = 1,
	/**
	\brief The trace settings are valid only for this instance.
	*/
	BGAPI_TRACESUPPORT_INSTANCE = 2,
	/**
	\brief Don't use that.
	*/
	BGAPI_TRACESUPPORT_FORCEINT = 0xFFFFFFFF
};

/**
\brief Trace option: insert timestamp.
*/
#define BGAPI_TRACEOUTPUTOPTION_TIMESTAMP			0x00000001
/**
\brief Trace option: insert timestamp difference.
*/
#define BGAPI_TRACEOUTPUTOPTION_TIMESTAMP_DIFF		0x00000002
/**
\brief Trace option: insert prefix.
*/
#define BGAPI_TRACEOUTPUTOPTION_PREFIX				0x00000100
/**
\brief Trace option: insert new line at the end of output string.
*/
#define BGAPI_TRACEOUTPUTOPTION_NEWLINE			0x00000200
/**
\brief Trace option: insert trace level.
*/
#define BGAPI_TRACEOUTPUTOPTION_TRACELEVEL			0x00000400

/**
\brief Trace mask: Output errors.
*/
#define BGAPI_TRACEMASK_ERROR			0x00000001
/**
\brief Trace mask: Output warnings.
*/
#define BGAPI_TRACEMASK_WARNING		0x00000002
/**
\brief Trace mask: Output informations.
*/
#define BGAPI_TRACEMASK_INFORMATION	0x00000004

/**
\brief Trace level 1
*/
#define BGAPI_TRACELEVEL_L1			0x00000010

/**
\brief Trace level 2
*/
#define BGAPI_TRACELEVEL_L2			0x00000020

/**
\brief Trace level 3
*/
#define BGAPI_TRACELEVEL_L3			0x00000040

/**
\brief Trace level 4
*/
#define BGAPI_TRACELEVEL_L4			0x00000080

/**
\brief Trace level all Bits
*/
#define BGAPI_TRACELEVEL_ALL		0x000000F0


/**
\brief Start pattern of BGAPI image header V1
*/
#define BGAPI_IMAGEHEADER_STARTPATTERN_V1		0x99992DD2

/**
\brief Start pattern of BGAPI image header V2
*/
#define BGAPI_IMAGEHEADER_STARTPATTERN_V2		0x1A2A3A4A

/**
\brief BGAPI Image header
*/
union BGAPI_ImageHeader
{
	/** 
	\brief Structure with start pattern of BGAPI Image Header  
	*/	
	struct {
		/**
		\brief 00
		\brief Start pattern of BGAPI Image Header. Compare this value with BGAPI_IMAGEHEADER_STARTPATTERN_V1 and BGAPI_IMAGEHEADER_STARTPATTERN_V2 defines, to deside which version is used. 
		*/
		unsigned int value;
		/**
		\brief 01-63
		\brief Not defined
		*/
		unsigned int uiNotDefined[63];
	}StartPattern;
	/** 
	\brief BGAPI Image Header V1  
	*/
	struct  {
		/**
		\brief 00
		\brief Start pattern of BGAPI Image Header V1  
		*/
		unsigned int uiStartPattern;
		/**
		\brief 01
		\brief hardware type id of camera
		*/
		unsigned int uiIdentifier;
		/**
		\brief 02
		\brief hardware counter of current frame
		*/
		unsigned int uiFrameCounter;
		/**
		\brief 03_0
		\brief content of camera configuration register 0
		*/
		unsigned short usConfigurationC0;
		/**
		\brief 03_1
		\brief content of camera configuration register 1
		*/
		unsigned short usConfigurationC1;
		/**
		\brief 04_0
		\brief start position x of current frame
		*/
		unsigned short usFrameStartPositionX;
		/**
		\brief 04_1
		\brief start position y of current frame
		*/
		unsigned short usFrameStartPositionY;
		/**
		\brief 05_0
		\brief extension in x direction of current frame
		*/
		unsigned short usFrameExtensionX;
		/**
		\brief 05_1
		\brief extension in y direction of current frame
		*/
		unsigned short usFrameExtensionY;
		/**
		\brief 06
		\brief user defined settings
		*/
		unsigned int uiUserDefined;
		/**
		\brief 07
		\brief current exposure time in I/O quality
		*/
		unsigned int uiExposure;
		/**
		\brief 08
		\brief current gain factor in I/O quality
		*/
		unsigned int uiGain;
		/**
		\brief 09_0
		\brief current color steering values for Gain0 in I/O quality
		*/
		unsigned char ucGain0;
		/**
		\brief 09_1
		\brief current color steering values for Gain1 in I/O quality
		*/
		unsigned char ucGain1;
		/**
		\brief 09_2
		\brief current color steering values for Gain2 in I/O quality
		*/
		unsigned char ucGain2;
		/**
		\brief 09_3
		\brief current color steering values for Gain3 in I/O quality
		*/
		unsigned char ucGain3;
		/**
		\brief 10
		\brief Status of camera
		*/
		unsigned int uiStatus;
		/**
		\brief 11_0
		\brief content of camera configuration register 2
		*/
		unsigned short usConfigurationExpC2;
		/**
		\brief 11_1
		\brief content of camera configuration register 3
		*/
		unsigned short usConfigurationExpC3;
		/**
		\brief 12-62
		\brief Not defined
		*/
		unsigned int uiNotDefinedV1[51];
		/**
		\brief 63
		\brief Stop pattern of BGAPI Image Header V1
		*/
		unsigned int uiStopPattern;		///< stop pattern of header see FF_STOPPATTERN
	}v1;
	/** 
	\brief BGAPI Image Header V2
	*/
	struct  {
		/**
		\brief 00
		\brief Start pattern of BGAPI Image Header V2  
		*/
		unsigned int uiStartPattern;
		/**
		\brief 01
		\brief hardware type id of camera
		*/
		unsigned int uiIdentifier;
		/**
		\brief 02
		\brief first master register
		*/
		unsigned int uiMaster0;
		/**
		\brief 03
		\brief second master register
		*/
		unsigned int uiMaster1;
		/**
		\brief 04
		\brief first format register
		*/
		unsigned int uiFormat0;
		/**
		\brief 05
		\brief second master register
		*/
		unsigned int uiFormat1;
		/**
		\brief 06
		\brief format configuration, e.g. binning, partial, ...
		*/
		unsigned int uiFormatConfig;
		/**
		\brief 07
		\brief pixel resolution
		*/
		unsigned int uiBitMode;
		/**
		\brief 08
		\brief exposure value
		*/
		unsigned int uiExposureValue;
		/**
		\brief 09
		\brief exposure control
		*/
		unsigned int uiExposureControl;
		/**
		\brief 10
		\brief analog gain
		*/
		unsigned int uiGain;
		/**
		\brief 11
		\brief analog offset
		*/
		unsigned int uiOffset;
		/**
		\brief 12
		\brief digital io state
		*/
		unsigned int uiDigitalIO;
		/**
		\brief 13
		\brief flash delay
		*/
		unsigned int uiFlashDelay;
		/**
		\brief 14
		\brief trigger mode settings
		*/
		unsigned int uiTrigger;
		/**
		\brief 15
		\brief trigger delay in us
		*/
		unsigned int uiTriggerDelay;
		/**
		\brief 16
		\brief test pattern settings
		*/
		unsigned int uiTestpattern;
		/**
		\brief 17
		\brief pixel gain red
		*/
		unsigned int uiPixelGainRed;
		/**
		\brief 18
		\brief pixel gain blue
		*/
		unsigned int uiPixelGainBlue;
		/**
		\brief 19
		\brief pixel gain green red
		*/
		unsigned int uiPixelGainGreenRed;
		/**
		\brief 20
		\brief pixel gain green blue
		*/
		unsigned int uiPixelGainGreenBlue;
		/**
		\brief 21_0
		\brief start position x of current frame
		*/
		unsigned short usFrameStartPositionX;
		/**
		\brief 21_1
		\brief start position y of current frame
		*/
		unsigned short usFrameStartPositionY;
		/**
		\brief 22_0
		\brief extension in x direction of current frame
		*/
		unsigned short usFrameExtensionX;
		/**
		\brief 22_1
		\brief extension in y direction of current frame
		*/
		unsigned short usFrameExtensionY;
		/**
		\brief 23
		\brief linecounter
		*/
		unsigned int uiLineCounter;
		/**
		\brief 24
		\brief sub frame length
		*/
		unsigned int uiSubFrameLength;
		/**
		\brief 25
		\brief framecounter
		*/
		unsigned int uiFrameCounter;
		/**
		\brief 26
		\brief user defined special register
		*/
		unsigned int uiUserDefined;
		/**
		\brief 27
		\brief Temperature in C
		*/
		unsigned int uiTemperature;
		/**
		\brief 28
		\brief sequencer configuration
		*/
		unsigned int uiSequencer;
		/**
		\brief 29
		\brief dataformat Rawbayer, RGB, YUV
		*/
		unsigned int uiDataFormat;
		/**
		\brief 30
		\brief timestamp for image capture
		*/
		unsigned int uiTimeStamp;
		/**
		\brief 31-62
		\brief Not defined
		*/
		unsigned int uiNotDefined[32];
		/**
		\brief 63
		\brief Stop pattern of BGAPI Image Header V2 
		*/
		unsigned int uiStopPattern;
	}v2;
};

/**
\brief BGAPI_Line
*/
enum BGAPI_Line
{
	/**
	\brief Line 0
	*/
	BGAPI_LINE_0 = 0,
	/**
	\brief Line 1
	*/
	BGAPI_LINE_1 = 1,
	/**
	\brief Line 2
	*/
	BGAPI_LINE_2 = 2,
	/**
	\brief Line 3
	*/
	BGAPI_LINE_3 = 3,
	/**
	\brief Line 4
	*/
	BGAPI_LINE_4 = 4,
	/**
	\brief Line 5
	*/
	BGAPI_LINE_5 = 5,
	/**
	\brief Line 6
	*/
	BGAPI_LINE_6 = 6,
	/**
	\brief Line 7
	*/
	BGAPI_LINE_7 = 7,
	/**
	\brief Line 8
	*/
	BGAPI_LINE_8 = 8,
	/**
	\brief Line 9
	*/
	BGAPI_LINE_9 = 9,
	/**
	\brief Line 10
	*/
	BGAPI_LINE_10 = 10,
	/**
	\brief Line 11
	*/
	BGAPI_LINE_11 = 11,
	/**
	\brief Line 12
	*/
	BGAPI_LINE_12 = 12,
	/**
	\brief Line 13
	*/
	BGAPI_LINE_13 = 13,
	/**
	\brief Line 14
	*/
	BGAPI_LINE_14 = 14,
	/**
	\brief Line 15
	*/
	BGAPI_LINE_15 = 15,
	/**
	\brief Internal line
	*/
	BGAPI_LINE_INTERNCLOCK = 64,
	/**
	\brief Don't use that.
	*/
	BGAPI_LINE_FORCEINT = 0xFFFFFFFF
};

/**
\brief BGAPI_UserOutput
*/
enum BGAPI_UserOutput
{
	/**
	\brief User Output 0
	*/
	BGAPI_USEROUTPUT_0 = 0,
	/**
	\brief User Output 1
	*/
	BGAPI_USEROUTPUT_1 = 1,
	/**
	\brief User Output 2
	*/
	BGAPI_USEROUTPUT_2 = 2,
	/**
	\brief User Output 3
	*/
	BGAPI_USEROUTPUT_3 = 3,
	/**
	\brief User Output 4
	*/
	BGAPI_USEROUTPUT_4 = 4,
	/**
	\brief User Output 5
	*/
	BGAPI_USEROUTPUT_5 = 5,
	/**
	\brief User Output 6
	*/
	BGAPI_USEROUTPUT_6 = 6,
	/**
	\brief User Output 7
	*/
	BGAPI_USEROUTPUT_7 = 7,
	/**
	\brief Don't use that.
	*/
	BGAPI_USEROUTPUT_FORCEINT = 0xFFFFFFFF
};

/**
\brief BGAPI_LineMode
*/
enum BGAPI_LineMode
{
	/**
	\brief This line is an output line.
	*/
	BGAPI_LINEMODE_OUTPUT = 0,
	/**
	\brief This line is an input line.
	*/
	BGAPI_LINEMODE_INPUT = 1,
	/**
	\brief Don't use that.
	*/
	BGAPI_LINEMODE_FORCEINT = 0xFFFFFFFF
};

/**
\brief BGAPI_LineOutputSource
*/
enum BGAPI_LineOutputSource
{
	/**
	\brief Set the output source to user output 0
	*/
	BGAPI_LINEOUTPUTSOURCE_USER0 = 0,
	/**
	\brief Set the output source to user output 1
	*/
	BGAPI_LINEOUTPUTSOURCE_USER1 = 1,
	/**
	\brief Set the output source to user output 2
	*/
	BGAPI_LINEOUTPUTSOURCE_USER2 = 2,
	/**
	\brief Set the output source to user output 3
	*/
	BGAPI_LINEOUTPUTSOURCE_USER3 = 3,
	/**
	\brief Set the output source to user output 4
	*/
	BGAPI_LINEOUTPUTSOURCE_USER4 = 4,
	/**
	\brief Set the output source to user output 5
	*/
	BGAPI_LINEOUTPUTSOURCE_USER5 = 5,
	/**
	\brief Set the output source to user output 6
	*/
	BGAPI_LINEOUTPUTSOURCE_USER6 = 6,
	/**
	\brief Set the output source to user output 7
	*/
	BGAPI_LINEOUTPUTSOURCE_USER7 = 7,
	/**
	\brief Set the output source to pwm 0
	*/
	BGAPI_LINEOUTPUTSOURCE_PWM0 = 32,
	/**
	\brief Set the output source to pwm 1
	*/
	BGAPI_LINEOUTPUTSOURCE_PWM1 = 33,
	/**
	\brief Set the output source to pwm 2
	*/
	BGAPI_LINEOUTPUTSOURCE_PWM2 = 34,
	/**
	\brief Set the output source to pwm 3
	*/
	BGAPI_LINEOUTPUTSOURCE_PWM3 = 35,	
	/**
	\brief Set the output source to Sequence Exposure
	*/	
	BGAPI_LINEOUTPUTSOURCE_SEQUENCEEXPOSURE = 90,
	/**
	\brief Set the output source to flash output
	*/
	BGAPI_LINEOUTPUTSOURCE_FLASH = 100,
	/**
	\brief Set the output source to readout active
	*/
	BGAPI_LINEOUTPUTSOURCE_READOUTACTIVE = 110,
	/**
	\brief Set the output source to trigger ready
	*/
	BGAPI_LINEOUTPUTSOURCE_TRIGGERREADY = 120,
	/**
	\brief Set the output source to trigger overlapped
	*/
	BGAPI_LINEOUTPUTSOURCE_TRIGGEROVERLAPPED = 121,
	/**
	\brief Set the output source to trigger skipped
	*/
	BGAPI_LINEOUTPUTSOURCE_TRIGGERSKIPPED = 122,
	/**
	\brief Loop back functionality.
	*/
	BGAPI_LINEOUTPUTSOURCE_LOOPBACK = 130,
	/**
	\brief Line 0.
	*/
	BGAPI_LINEOUTPUTSOURCE_LINE0 = 131,
	/**
	\brief Line 1.
	*/
	BGAPI_LINEOUTPUTSOURCE_LINE1 = 132,
	/**
	\brief Line 2.
	*/
	BGAPI_LINEOUTPUTSOURCE_LINE2 = 133,
	/**
	\brief Line 3.
	*/
	BGAPI_LINEOUTPUTSOURCE_LINE3 = 134,
	/**
	\brief Disable output source.
	*/
	BGAPI_LINEOUTPUTSOURCE_OFF = 256,
	/**
	\brief Don't use that.
	*/
	BGAPI_LINEOUTPUTSOURCE_FORCEINT = 0xFFFFFFFF
};

/**
\brief Enumeration to control the line output gate
*/
enum BGAPI_LineGate
{
	/**
	\brief Lock the output signal.
	*/
	BGAPI_LINEGATE_OFF = 0,
	/**
	\brief Let pass the output signal .
	*/
	BGAPI_LINEGATE_ON = 1,
	/**
	\brief Use the output signal as retrigger.
	*/
	BGAPI_LINEGATE_RETRIGGER = 2,
	/**
	\brief Don't use that.
	*/
	BGAPI_LINEGATE_FORCEINT = 0xFFFFFFFF
};

/**
\brief BGAPI_PWM
*/
enum BGAPI_PWM
{
	/**
	\brief pwm 0
	*/
	BGAPI_PWM_0 = 0,
	/**
	\brief pwm 1
	*/
	BGAPI_PWM_1 = 1,
	/**
	\brief pwm 2
	*/
	BGAPI_PWM_2 = 2,
	/**
	\brief pwm 3
	*/
	BGAPI_PWM_3 = 3,
	/**
	\brief pwm 3
	*/
	BGAPI_PWM_4 = 4,
	/**
	\brief pwm 3
	*/
	BGAPI_PWM_5 = 5,
	/**
	\brief pwm 3
	*/
	BGAPI_PWM_6 = 6,
	/**
	\brief pwm 3
	*/
	BGAPI_PWM_7 = 7,
	/**
	\brief Don't use that.
	*/
	BGAPI_PWM_FORCEINT = 0xFFFFFFFF
};

/**
\brief Hardware based timer
*/
enum BGAPI_Timer
{
	/**
	\brief Timer 1
	*/
	BGAPI_TIMER_1 = 1,
	/**
	\brief Timer 2
	*/
	BGAPI_TIMER_2 = 2,
	/**
	\brief Timer 3
	*/
	BGAPI_TIMER_3 = 3,
	/**
	\brief Timer 4
	*/
	BGAPI_TIMER_4 = 4,
	/**
	\brief Timer 5
	*/
	BGAPI_TIMER_5 = 5,
	/**
	\brief Timer 6
	*/
	BGAPI_TIMER_6 = 6,
	/**
	\brief Timer 7
	*/
	BGAPI_TIMER_7 = 7,
	/**
	\brief Timer 8
	*/
	BGAPI_TIMER_8 = 8,
	/**
	\brief Don't use that.
	*/
	BGAPI_TIMER_FORCEINT = 0xFFFFFFFF
};

/**
\brief Hardware based counter
*/
enum BGAPI_Counter
{
	/**
	\brief Counter 1
	*/
	BGAPI_COUNTER_1 = 1,
	/**
	\brief Counter 2
	*/
	BGAPI_COUNTER_2 = 2,
	/**
	\brief Counter 3
	*/
	BGAPI_COUNTER_3 = 3,
	/**
	\brief Counter 4
	*/
	BGAPI_COUNTER_4 = 4,
	/**
	\brief Counter 5
	*/
	BGAPI_COUNTER_5 = 5,
	/**
	\brief Counter 6
	*/
	BGAPI_COUNTER_6 = 6,
	/**
	\brief Counter 7
	*/
	BGAPI_COUNTER_7 = 7,
	/**
	\brief Counter 8
	*/
	BGAPI_COUNTER_8 = 8,
	/**
	\brief Don't use that.
	*/
	BGAPI_COUNTER_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_MessageId defines several kinds of messages, which can be signaled asynchronous via message callback.
*/
enum BGAPI_MessageId
{
	/**
	\brief Enables all defined messages. Currently not supported.
	*/
	BGAPI_MESSAGEID_ALL = 0,
	/**
	\brief Enables all line messages. Currently not supported.
	\brief This message signals a change in state and the current state of one or more digital inputs.
	*/
	BGAPI_MESSAGEID_IOALL = 1,
	
	/**
	\brief Enables error messages. For GigE cameras this id enables GigEVision error codes.
	*/
	BGAPI_MESSAGEID_ERROR = 10,
	/**
	\brief Event lost.
	*/
	BGAPI_MESSAGEID_EVENTLOST = 11,
	/**
	\brief Temperature exceeded.
	*/
	BGAPI_MESSAGEID_TEMPERATUREEXCEEDED = 12,
	/**
	\brief Heartbeat timeout.
	*/
	BGAPI_MESSAGEID_HEARTBEATTIMEOUT = 13,
	/**
	\brief Trigger overlapped.
	*/
	BGAPI_MESSAGEID_TRIGGEROVERLAPPED = 14,
	/**
	\brief Trigger skipped.
	*/
	BGAPI_MESSAGEID_TRIGGERSKIPPED = 15,
	/**
	\brief Exposure start.
	*/
	BGAPI_MESSAGEID_EXPOSURESTART = 16,
	/**
	\brief Exposure end.
	*/
	BGAPI_MESSAGEID_EXPOSUREEND = 17,
	/**
	\brief Trigger ready.
	*/
	BGAPI_MESSAGEID_TRIGGERREADY = 18,
	/**
	\brief Frame start.
	*/
	BGAPI_MESSAGEID_FRAMESTART = 19,
	/**
	\brief Frame end.
	*/
	BGAPI_MESSAGEID_FRAMEEND = 20,
	/**
	\brief Line 0 rising edge.
	*/
	BGAPI_MESSAGEID_LINE0RISINGEDGE = 21,
	/**
	\brief Line 0 falling edge.
	*/
	BGAPI_MESSAGEID_LINE0FALLINGEDGE = 22,
	/**
	\brief Line 1 rising edge.
	*/
	BGAPI_MESSAGEID_LINE1RISINGEDGE = 23,
	/**
	\brief Line 1 falling edge.
	*/
	BGAPI_MESSAGEID_LINE1FALLINGEDGE = 24,
	/**
	\brief Line 2 rising edge.
	*/
	BGAPI_MESSAGEID_LINE2RISINGEDGE = 25,
	/**
	\brief Line 2 falling edge.
	*/
	BGAPI_MESSAGEID_LINE2FALLINGEDGE = 26,
	/**
	\brief Line 3 rising edge.
	*/
	BGAPI_MESSAGEID_LINE3RISINGEDGE = 27,
	/**
	\brief Line 3 falling edge.
	*/
	BGAPI_MESSAGEID_LINE3FALLINGEDGE = 28,
	/**
	\brief Line 4 rising edge.
	*/
	BGAPI_MESSAGEID_LINE4RISINGEDGE = 29,
	/**
	\brief Line 4 falling edge.
	*/
	BGAPI_MESSAGEID_LINE4FALLINGEDGE = 30,
	/**
	\brief Line 5 rising edge.
	*/
	BGAPI_MESSAGEID_LINE5RISINGEDGE = 31,
	/**
	\brief Line 5 falling edge.
	*/
	BGAPI_MESSAGEID_LINE5FALLINGEDGE = 32,

	/**
	\brief End of last exposure of a sequence.
	*/
	BGAPI_MESSAGEID_ENDOFSEQUENCEEXPOSURE = 33,

	/**
	\brief Over power threshold of PWM 0.
	*/
	BGAPI_MESSAGEID_PWMOVERPOWER0 = 34,

	/**
	\brief Over power threshold of PWM 1.
	*/
	BGAPI_MESSAGEID_PWMOVERPOWER1 = 35,

	/**
	\brief Over power threshold of PWM 2.
	*/
	BGAPI_MESSAGEID_PWMOVERPOWER2 = 36,

	/**
	\brief Over power threshold of PWM 3.
	*/
	BGAPI_MESSAGEID_PWMOVERPOWER3 = 37,

	/**
	\brief Don't use that.
	*/
	BGAPI_MESSAGEID_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_MessageState defines possible message states.
*/
enum BGAPI_MessageState
{
	/**
	\brief Deactivate a messages.
	*/
	BGAPI_MESSAGESTATE_OFF = 0,
	/**
	\brief Activate a message.
	*/
	BGAPI_MESSAGESTATE_ON = 1,
	/**
	\brief Activate a message with appending data block
	*/
	BGAPI_MESSAGESTATE_DATA = 2,
	/**
	\brief Don't use that.
	*/
	BGAPI_MESSAGESTATE_FORCEINT = 0xFFFFFFFF
};

/**
\brief The enumeration BGAPI_LinePotential defines possible line potential.
*/
enum BGAPI_LinePotential
{
	/**
	\brief The line potential is intern.
	*/
	BGAPI_LINEPOTENTIAL_INTERN = 0,
	/**
	\brief the line potential is extern.
	*/
	BGAPI_LINEPOTENTIAL_EXTERN = 1,
	/**
	\brief Don't use that.
	*/
	BGAPI_LINEPOTENTIAL_FORCEINT = 0xFFFFFFFF
};

typedef struct
{
	unsigned int uiDataLength;		//Length of data block
	unsigned short usCounter;		//Message counter
	unsigned short usId;			//Message Id (see enumeration BGAPI_MessageId)
	unsigned int uiTimestampHigh;	//Timestamp of message hight part
	unsigned int uiTimestampLow;	//Timestamp of message low part
	union DATA
	{
		//use this struct if usId == BGAPI_MESSAGEID_IOALL
		struct IOALL
		{
			unsigned short usInputState;	//bit field for the current state of digital inputs
			unsigned short usInputChanged;	//bit field for change in state of digital inputs
			unsigned int uiNotDefined[139];	//reserved
		}ioall;
		//use this element if usId == BGAPI_MESSAGEID_ALL
		struct ALL
		{
			unsigned int uiNotDefined[140];
		}all;
		
	}data;
} BGAPI_Message;



/**
\brief The enumeration BGAPI_PnPEvent defines several kinds of plug and play events.
*/
enum BGAPI_PnPEvent
{
	/**
	\brief Device was removed.
	*/
	BGAPI_PNPEVENT_DEVICEREMOVED = 0,
	/**
	\brief Device was added.
	*/
	BGAPI_PNPEVENT_DEVICEADD = 1,
	/**
	\brief Remove a camera attached hdmi device
	*/
	BGAPI_PNPEVENT_DEVICEHDMIREMOVED = 2,
	/**
	\brief Add a camera attached hdmi device
	*/
	BGAPI_PNPEVENT_DEVICEHDMIADD = 3,
	/**
	\brief Don't use that.
	*/
	BGAPI_PNPEVENT_FORCEINT = 0xFFFFFFFF
};

/**
\brief Chunk id's
*/
enum BGAPI_ChunkId
{
	/**
	\brief Image data
	*/
	BGAPI_CHUNKID_IMAGE = 1,
	
	/**
	\brief Header data
	*/
	BGAPI_CHUNKID_HEADER = 2,

	/**
	\brief Distance information of 3D cameras
	*/
	BGAPI_CHUNKID_DISTANCE = 3,

	/**
	\brief Don't use that.
	*/
	BGAPI_CHUNKID_FORCEINT = 0xFFFFFFFF
};

/**
\brief BGAPI_ProcessingUnit
*/
enum BGAPI_ProcessingUnit
{
	/**
	\brief The operation works on hardware.
	*/
	BGAPI_PROCESSINGUNIT_HARDWARE = 0,
	/**
	\brief The operation works in software.
	*/
	BGAPI_PROCESSINGUNIT_SOFTWARE = 1,
	/**
	\brief Don't use that.
	*/
	BGAPI_PROCESSINGUNIT_FORCEINT = 0xFFFFFFFF
};

/**
\brief BGAPI_AutomaticMode
*/
enum BGAPI_AutomaticMode
{
	/**
	\brief The automatic mode is off.
	*/
	BGAPI_AUTOMATICMODE_OFF = 0,
	/**
	\brief The automatic adjustment is performed once.
	*/
	BGAPI_AUTOMATICMODE_ONCE = 1,
	/**
	\brief The automatic adjusment is continuously done.
	*/
	BGAPI_AUTOMATICMODE_CONTINUOUS = 2,
	/**
	\brief Don't use that.
	*/
	BGAPI_AUTOMATICMODE_FORCEINT = 0xFFFFFFFF
};

/**
\brief BGAPI_HDRParameter
*/
enum BGAPI_HDRParameter
{
	/**
	\brief Exposure.
	*/
	BGAPI_HDRPARAMETER_EXPOSURE = 0,
	/**
	\brief Percent.
	*/
	BGAPI_HDRPARAMETER_PERCENT = 1,
	/**
	\brief Don't use that.
	*/
	BGAPI_HDRPARAMETER_FORCEINT = 0xFFFFFFFF
};
#endif //_BGAPIDEF_HPP__
