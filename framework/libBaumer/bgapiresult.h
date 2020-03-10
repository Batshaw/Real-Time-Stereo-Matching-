/*****************************************************************************\
*    Program:		bgapi
*    Version:		$Revision: 7997 $	
*    Module:		$Source$
*    Compiler:		
*    Operating System:
*    Purpose:
*
*    Notes:
*    Created:		17.04.2007
*    Last modified:	$Date: 2009-07-22 11:09:56 +0200 (Mi, 22 Jul 2009) $
*                         
*                         
*    Author:		Silvio Voitzsch
*	 Last $Author: svo $
*    Copyright:		Baumer Optronic GmbH, Radeberg
*
*	 $Log: bgapiresult.h,v $
*	 Revision 1.4  2007/11/27 09:15:23  svo
*	 file header added
*	
\*****************************************************************************/

#ifndef _BGAPIRESULT_H__
#define _BGAPIRESULT_H__
//Baumer Optronic
//bgapi Interface
//Error codes / warnings

//bgapi return value
typedef int BGAPI_RESULT;


/**
\brief Operation OK.
*/
#define BGAPI_RESULT_OK                      1

/**
\brief General information: The operation can not be called twice.
*/
#define BGAPI_RESULT_ALREADYDONE			2

/**
\brief General error: The operation failed. For detailed informations enable trace output.
*/
#define BGAPI_RESULT_FAIL                    0

/**
\brief General error: No memory allocated.
\brief The most frequent case for this error is, if the user does not set image memory, when the Image object was created with 'external_buffer' option. In this case the function getImage return this error (see functions Image::set, Image::get, System::createImage).
*/
#define BGAPI_RESULT_FAIL_MEMORY             -1

/**
\brief General error: The passed parameter is invalid.
\brief The most frequent case for this error is, when using the structures with prefix BGAPIX_. The member cbSize have to filled before using it (Only for C/C++ programmers).
\brief Second case: The byte order does not agree. Please check your compiler options which affect the byte order of structure.
*/
#define BGAPI_RESULT_FAIL_INVALID_PARAM      -2

/**
\brief General error: The operation is not supported.
*/
#define BGAPI_RESULT_NOT_SUPPORTED			-3

/**
\brief The BGAPI object is not valid. 
*/
#define BGAPI_RESULT_FAIL_NOBGAPIOBJECT			-4

/**
\brief System error: You passed an invalid System index to function createSystem.
*/
#define BGAPI_RESULT_SYSTEM_INVALID_ID		-21

/**
\brief System error: The System was not correctly created.
*/
#define BGAPI_RESULT_SYSTEM_UNUSED			-22

/**
\brief System error: Returned by all System functions, if the System::open function returned with error.
*/
#define BGAPI_RESULT_SYSTEM_NOTOPEN			-23

/**
\brief System error: Error in device module while creating a system. The device module was not bound to the bgapi System.
*/
#define BGAPI_RESULT_SYSTEM_NOTBOUND			-24

/**
\brief System error: Release and recreate the system to use this function again.
*/
#define BGAPI_RESULT_SYSTEM_FUNCTIONNOTAVAILABLE	-25

/**
\brief System error: System not created.
*/
#define BGAPI_RESULT_SYSTEM_NOTCREATE	-26

/**
\brief Camera error: You passed an invalid Camera index to function createCamera.
*/
#define BGAPI_RESULT_CAMERA_INVALID_ID       -30

/**
\brief Camera error: The Camera was not completley created. Wrong camera configuration, e.g. wrong ip configuration for GigE devices.
*/
#define BGAPI_RESULT_CAMERA_UNUSEABLE           -31

/**
\brief Camera error: Error in device module while creating a camera. The device module was not bound to the bgapi Camera.
*/
#define BGAPI_RESULT_CAMERA_NOTBOUND			-34

/**
\brief Camera error: Returned by all Camera functions, if function createCamera returned with error.
*/
#define BGAPI_RESULT_CAMERA_NOTCREATE		-36

/**
\brief Camera error: Returned by all Camera functions, if function bgapi.Camera.open returned with error.
*/
#define BGAPI_RESULT_CAMERA_NOTOPEN		-37

/**
\brief Camera error: Returned by Camera object in queued data access mode, when trying to release an image object, while the camera is running. Please stop the camera first. See Camera.setStart function.
*/
#define BGAPI_RESULT_CAMERA_RUNNING		-38

/**
\brief Camera is busy.
*/
#define BGAPI_RESULT_CAMERA_BUSY		-39

/**
\brief Feature error: The feature is not supported by the bgapi object.
*/
#define BGAPI_RESULT_FEATURE_NOTSUPPORTED	-50

/**
\brief Feature error: Error while creating the feature tree.
*/
#define BGAPI_RESULT_FEATURE_INVALID			-51

/**
\brief Feature error: No next feature available.
*/
#define BGAPI_RESULT_FEATURE_NONEXT			-52

/**
\brief Feature error: No sub feature available.
*/
#define BGAPI_RESULT_FEATURE_NOSUB			-53

/**
\brief Feature error: Feature supports no value (see getValue).
*/
#define BGAPI_RESULT_FEATURE_NOVALUE			-54

/**
\brief Feature error: Feature is not available.
*/
#define BGAPI_RESULT_FEATURE_NOTAVAILABLE	-55

/**
\brief Feature error: Feature is not implemented.
*/
#define BGAPI_RESULT_FEATURE_NOTIMPLEMENTED	-56

/**
\brief Feature error: Low level error in device module.
*/
#define BGAPI_RESULT_FEATURE_FAIL			-57

/**
\brief Feature error: Feature is locked.
*/
#define BGAPI_RESULT_FEATURE_LOCKED			-58

/**
\brief Feature error: The feature refreshing does not work correctly (getFeature).
*/
#define BGAPI_RESULT_FEATURE_REFRESH		-59

/**
\brief Feature error: The feature was accessed invalid. Please Check the accessmode if you are not sure about accessing features.
*/
#define BGAPI_RESULT_FEATURE_WRONGACCESSMODE		-60

/**
\brief Value error: You have used the wrong function. The value has an other type (see BGAPI_Type)
*/
#define BGAPI_RESULT_VALUE_WRONGTYPE			-70

/**
\brief Value error: No data memory was allocated.
*/
#define BGAPI_RESULT_VALUE_MEMORY			-72

/**
\brief Value error: The value does not exist.
*/
#define BGAPI_RESULT_VALUE_INEXISTENT	-73

/**
\brief Image error: Error in image module while creating a image. The image module was not bound to the bgapi Image.
*/
#define BGAPI_RESULT_IMAGE_NOTBOUND			-100

/**
\brief Image error: The image object is assigned to a camera object. To deallocate use the function Camera.freeImage.
*/
#define BGAPI_RESULT_IMAGE_INUSE			-101

/**
\brief Image error: The image object is already released.
*/
#define BGAPI_RESULT_IMAGE_UNUSABLE			-102

#endif //_BGAPIRESULT_H__
