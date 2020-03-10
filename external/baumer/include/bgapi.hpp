/*****************************************************************************\
*    Program:		bgapi
*    Version:		$Revision: 8500 $	
*    Module:		$Source$
*    Compiler:		
*    Operating System:
*    Purpose:
*
*    Notes:
*    Created:		17.04.2007
*    Last modified:	$Date: 2009-10-28 09:08:13 +0100 (Mi, 28 Okt 2009) $
*                         
*                         
*    Author:		Silvio Voitzsch
*	 Last $Author: oth $
*    Copyright:		Baumer Optronic GmbH, Radeberg
*
*	 $Log: bgapi.hpp,v $
*	 Revision 1.28  2008/04/08 08:52:10  svo
*	 io's and message channel
*	
*	 Revision 1.27  2008/04/07 06:16:43  svo
*	 io's and message channel
*	
*	 Revision 1.26  2008/03/13 10:02:29  svo
*	 new feature TRIGGERSYNCSW
*	
*	 Revision 1.25  2008/02/22 13:15:11  svo
*	 BGAPI_ImageHeader defined
*	
*	 Revision 1.24  2008/02/18 10:51:22  svo
*	 getReadoutTime added und basic feature id BGAPI_FEATUREID_CAMERA_READOUTTIME
*	
*	 Revision 1.23  2008/01/22 10:35:20  svo
*	 *** empty log message ***
*	
*	 Revision 1.22  2008/01/21 07:31:12  svo
*	 declaration of image class changed; new function doTransform; functions doScale, doFilter and doDemosaic deleted.
*	
*	 Revision 1.21  2007/12/20 15:31:19  svo
*	 *** empty log message ***
*	
*	 Revision 1.20  2007/12/12 07:51:23  svo
*	 getSequencerAddParameter new parameter pnumofparamsets added
*	
*	 Revision 1.19  2007/12/06 15:05:52  svo
*	 parameter of function doSequencerAddParameter removed
*	
*	 Revision 1.18  2007/12/06 09:24:50  svo
*	 new parameter for function doSequencerAddParameter
*	
*	 Revision 1.17  2007/12/06 07:41:03  svo
*	 bo standard sequencer defined
*	
*	 Revision 1.16  2007/11/27 09:15:16  svo
*	 file header added
*	
\*****************************************************************************/

/** 
\file bgapi.hpp
\brief Baumer Optronic bgapi
\brief C++ - Interface Definition.
*/ 

#ifndef _BGAPI_HPP__
#define _BGAPI_HPP__

#include "bgapidef.hpp"
#define _BGAPIDEF_H__
#include "bgapi.h"
#undef _BGAPIDEF_H__

namespace BGAPI
{
class System;
class Camera;
class Image;
class Feature;
class Value;
class TraceObject;

/**
\brief Image notification function pointer. 
*/
typedef void (BGAPI_CALLBACK *BGAPI_NOTIFY_CALLBACK)( void * callBackOwner, Image * image );

/**
\brief Message notification function pointer. 
*/
typedef void (BGAPI_CALLBACK *BGAPI_MESSAGE_CALLBACK)( void * callBackOwner, BGAPI_Message * message );

/**
\brief Trace notification function pointer.
*/
typedef void (BGAPI_CALLBACK *BGAPI_TRACE_CALLBACK)( void * callBackOwner, char ** tracemsg );

/**
\brief Plug and Play notification function pointer.
*/
typedef void (BGAPI_CALLBACK *BGAPI_PNP_CALLBACK)( void * callBackOwner, BGAPI_PnPEvent * pnpevent );

/**
\brief Bus Reset notification function pointer.
*/
typedef void (BGAPI_CALLBACK *BGAPI_BUSRESET_CALLBACK)( void * callBackOwner );

/**
\fn BGAPI_RESULT countSystems( int * count )
\brief Function category: Entry Point.
\brief Count the numer of available BGAPI systems.
\param 'count' (OUT) The numer of available BGAPI systems.
\retval 'BGAPI_RESULT_OK' No error.
*/
inline BGAPI_RESULT countSystems( int * count ) { return BGAPI_countSystems( count ); }

/**
\fn BGAPI_RESULT createSystem( int index, System **system )
\brief Function category: Entry Point.
\brief Use this function to create an instance of BGAPI system. Is the 'index' larger than or equal to the count of found BGAPI systems, the function failes.
\param 'index' (in) The index of the BGAPI system to create.
\param 'system' (OUT) The pointer to the created BGAPI system.
*/
inline BGAPI_RESULT createSystem( int index, System **system ) { return BGAPI_createSystem( index, (BGAPI_System **)system); }

/**
\fn BGAPI_RESULT createImage( Image ** img )
\brief Function category: Entry Point.
\brief Use this function to create an instance of BGAPI image.
\param 'img' (OUT) The pointer to the BGAPI image.
\retval 'BGAPI_RESULT_OK'	No error.
*/
inline BGAPI_RESULT createImage( Image ** img ) { return BGAPI_createImage( (BGAPI_Image **)img ); }

/**
\fn BGAPI_RESULT releaseImage( Image * img )
\brief Function category: Entry Point.
\brief Release a BGAPI image object. This function destroys the BGAPI Image object and frees all used resources. 
\brief When using external allocated buffer via function 'Image.setBuffer' the buffer remains untouched by BGAPI.
\param 'img' (IN) BGAPI image object to release.
\retval 'BGAPI_RESULT_OK'	No error.
*/
inline BGAPI_RESULT releaseImage( Image * img ) { return BGAPI_releaseImage( (BGAPI_Image*)img ); }

/**
\fn BGAPI_RESULT getBGAPIVersionLength( int * length )
\brief Function category: Entry Point.
\brief Use this function to get the version of BGAPI.
\param 'length' (OUT) The length of BGAPI version string.
\retval 'BGAPI_RESULT_OK'	No error.
*/
inline BGAPI_RESULT getBGAPIVersionLength( int * length ) { return BGAPI_getBGAPIVersionLength( length ); }

/**
\fn BGAPI_RESULT getBGAPIVersion( char * bgapiversion, int length )
\brief Function category: Entry Point.
\brief Use this function to get the version string of BGAPI.
\param 'bgapiversion' (OUT) The version string of BGAPI.
\param 'length' (IN) The length of BGAPI version string, delivered by function 'getBGAPIVersionLength'.
\retval 'BGAPI_RESULT_OK'	No error.
*/
inline BGAPI_RESULT getBGAPIVersion( char * bgapiversion, int length ) { return BGAPI_getBGAPIVersion( bgapiversion, length ); }

/**
\fn BGAPI_RESULT setModuleSearchPath( char * path, int length )
\brief Function category: Entry Point.
\brief Use this function to set the search path of BGAPI modules, including system modules and image module like the BGAPI img_proc library.
\brief In standard BGAPI searchs in current working directory and in directories which are defined in environment variable 'PATH'.
\brief When using this function, BGAPI only seacrchs in the specified dirctory.
\param 'path' (IN) The new search directory.
\param 'length' (IN) The length of 'path'.
\retval 'BGAPI_RESULT_OK'	No error.
*/
inline BGAPI_RESULT setModuleSearchPath( char * path, int length ) {return BGAPI_setModuleSearchPath( path, length ); }

/**
\brief Representation of a BGAPI system object.
*/
class BGAPI_DECL System
{
private:
	/**
	\brief Function category: Constructor.
	\brief The constructor is a private function and can not directly called. To create an instance of this object use function BGAPI.createSystem.
	*/
	System();
    System( const System &Obj );
public:
	/**
	\brief Function category: Working with BGAPI objects.
	\brief This function opens a BGAPI system object. Only after opening the features of BGAPI system can be used.
	\retval 'BGAPI_RESULT_OK'	No error.
	\retval 'BGAPI_RESULT_SYSTEM_NOTOPEN' Error while opening the physical system device.
	*/
    BGAPI_RESULT open();

	/**
	\brief Function category: Working with BGAPI objects.
	\brief This function releases the BGAPI system object and frees all used resources.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT release();

	/**
	\brief Function category: Working with BGAPI objects.
	\brief This function count the connected cameras of the BGAPI system.
	\param 'count' (OUT) Get the count of connected cameras.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT countCameras( int* count );

	/**
	\brief Function category: Working with BGAPI objects.
	\brief Creates a camera object. Is the index larger than or equal to the count of connected cameras, the function failes.
	\param 'index' (IN) The index of BGAPI camera to create.
	\param 'cam' (OUT) The pointer to the created BGAPI camera.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT createCamera( int index, Camera ** cam );

	/**
	\brief Function category: Working with BGAPI objects.
	\brief This function releases the BGAPI camera object and frees all used resources.
	\param 'cam' (IN) The BGAPI camera object to release.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT releaseCamera( Camera * cam );

	/**
	\brief Function category: Working with BGAPI objects.
	\brief Deprecated, please use function BGAPI.createImage.
	\param 'img' (OUT) The pointer to the BGAPI image.
	\param 'useexternbuffer' (IN) Pass true, to use your own image memory (external buffer). Pass false, BGAPI allocates the image memory.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT createImage( Image ** img, bool useexternbuffer );

	/**
	\brief Function category: Working with BGAPI objects.
	\brief Deprecated, please use function BGAPI.releaseImage.
	\param 'img' (IN) BGAPI image object to release.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT releaseImage( Image * img );

	/**
	\brief Function category: BO Basic API.
	\brief This function returns the count of supported features of BGAPI system.
	\param 'count' (OUT) The count of BGAPI system features.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT getFeatureCount( int* count );

	/**
	\brief Function category: BO Basic API.
	\brief This function executes a basic API feature.
	\param 'toset' (IN) The basic feature to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT setFeature( Feature * toset );

	/**
	\brief Function category: BO Basic API.
	\brief This function refreshs the basic API feature state and returns the actual settings.
	\param 'Id' (IN) Identifier of basic API feature to refresh.
	\param 'cur' (OUT) The pointer to desired basic API feature.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT getFeature( BGAPI_FeatureId Id, Feature ** cur );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns current BGAPI system informations.
	\param 'pstate' (OUT) Current feature state.
	\param 'deviceinfo' (OUT) BGAPI system information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getDeviceInformation( BGAPI_FeatureState * pstate, BGAPIX_SystemInfo * deviceinfo ); 

	/**
	\brief Function category: BO Extended API.
	\brief This function sets the GVS driver model.
	\param 'drivermodel' (IN) The driver model to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setGVSDriverModel( BGAPI_GVSDriverModel drivermodel );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of feature GVSDriverModel and a list of supported driver models (see BGAPI_GVSDriverModel ).
	\param 'pstate' (OUT) Current feature state.
	\param 'list' (OUT) Current driver model and a list with supported driver models.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGVSDriverModel( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );

	/**
	\brief Function category: BO Extended API.
	\brief Initiate an action command. This command sends a broadcast action command to one or more receivers, like BGAPI cameras or trigger device.
	\param 'devicekey' (IN) A unique device key. All cameras with the identical device key accepts the action command. Please refer to function Camera.setActionCommand.
	\param 'groupkey' (IN) A unique group key to build camera groups.
	\param 'groupmask' (IN) For every group key a 32 bit group mask can be specified.
	\param 'bWaitForAnswer' (IN) If 'bWaitForAnswer' is set this function waits of the acknowledge of all registerd responders. Please use the function System.doCommandResponderListAdd to register a responder. Otherwise the function returns immediately.
	\retval 'BGAPI_RESULT_OK' No error.
	*/	
	BGAPI_RESULT doActionCommand( unsigned int devicekey, unsigned int groupkey, unsigned int groupmask, bool bWaitForAnswer );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of action command feature.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK' No error.
	*/	
	BGAPI_RESULT getActionCommand( BGAPI_FeatureState * pstate );
	
	/**
	\brief Function category: BO Extended API. Special Gige feature.
	\brief This function enables or disables the persistent ip mode for a Gige camera. When enable this feature you have to pass the ip address, subnet mask and gateway. In other case these parametere are ignored. To apply these settings reconnect the power suply of camera.
	\param 'mac' (IN) MAC address of the desired camera.
	\param 'ip' (IN) IP address to force.
	\param 'subnetmask' (IN) subnet mask to force.
	\param 'gateway' (IN) gateway to force.
	\param 'size' (IN) Count of used elements in the arrays ip, subnetmask and gateway.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT doGVSForceIP( unsigned char mac[6], unsigned char ip[32], unsigned char subnetmask[32], unsigned char gateway[32], int size );

	/**
	\brief Function category: BO Extended API.
	\brief If a action command with acknowledge is send, the BGAPI system must know all devices, which should respond the action command.
	\brief These devices are stored in a responder list. If one device does not reponse, the action command failes.
	\param 'device' (IN) The responding device. For GigE devices this value needs IPV4 address.
	\retval 'BGAPI_RESULT_OK' No error.
	*/	
	BGAPI_RESULT doCommandResponderListAdd( unsigned char device[32] );

	/**
	\brief Function category: BO Extended API.
	\brief Reset the action command responder list.
	\retval 'BGAPI_RESULT_OK' No error.
	*/	
	BGAPI_RESULT doCommandResponderListReset();

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of action command feature and the number of devices which have not responded.
	\param 'pstate' (OUT) Current feature state.
	\param 'tNumberOf' (OUT) The number of devices, which have not requested the system command specified by 'commandid'.
	\retval 'BGAPI_RESULT_OK' No error.
	*/		
	BGAPI_RESULT getNumberOfFailedCommandResponder( BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tNumberOf);

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of action command feature and a device, which have not responded a action command.
	\param 'pstate' (OUT) Current feature state.
	\param 'index' (IN) The index lay between 0 and the parameter 'tNumberOf' of function System.getNumberOfFailedCommandResponder.
	\param 'tDevice' (OUT) The device, which have not responded.
	\retval 'BGAPI_RESULT_OK' No error.
	*/		
	BGAPI_RESULT getFailedCommandResponder( BGAPI_FeatureState * pstate, int index, BGAPIX_TypeArrayBYTE * tDevice );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the number of available interfaces (e.g. network adapters). 
	\param 'pstate' (OUT) Current feature state.
	\param 'tNumberOf' (OUT) The number of available inerfaces (e.g. network adapters).
	\retval 'BGAPI_RESULT_OK' No error.
	*/		
	BGAPI_RESULT getNumberOfAvailableInterfaces( BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tNumberOf );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and a interface (e.g. network adapter), specified by 'index'. 
	\param 'index' (IN) Index of interface.
	\param 'pstate' (OUT) Current feature state.
	\param 'tInterface' (OUT) The interface at 'index'.
	\param 'tMask' (OUT) The subnet of the interface at 'index'.
	\retval 'BGAPI_RESULT_OK' No error.
	*/		
	BGAPI_RESULT getAvailableInterface( int index, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayBYTE * tInterface, BGAPIX_TypeArrayBYTE * tMask );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the number of interfaces (e.g. network adapters), which were ignored by searching of cameras. 
	\param 'pstate' (OUT) Current feature state.
	\param 'tNumberOf' (OUT) The number of ignored interfaces.
	\retval 'BGAPI_RESULT_OK' No error.
	*/		
	BGAPI_RESULT getNumberOfIgnoredInterfaces( BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tNumberOf );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and a ignored interface (e.g. network adapter), specified by 'index'. 
	\param 'index' (IN) Index of ignored interface.
	\param 'pstate' (OUT) Current feature state.
	\param 'tInterface' (OUT) The ignored interfaces at 'index'.
	\retval 'BGAPI_RESULT_OK' No error.
	*/		
	BGAPI_RESULT getIgnoredInterface( int index, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayBYTE * tInterface );

	/**
	\brief Function category: BO Extended API.
	\brief Exclude an interface (e.g. network adapter), by searching for cameras. 
	\param 'interfacetoignore' (IN) The interface to exclude. For GigE devices this value needs IPV4 address.
	\retval 'BGAPI_RESULT_OK' No error.
	*/		
	BGAPI_RESULT setInterfaceToIgnore( unsigned char interfacetoignore[32] );

	/**
	\brief Function category: BO Extended API.
	\brief Reset the list of ignored interfaces (e.g. network adapters). 
	\retval 'BGAPI_RESULT_OK' No error.
	*/		
	BGAPI_RESULT doInterfaceToIgnoreListReset();

	/**
	\brief Function category: Trace support.
	\brief This function creates a trace object to log some internal processes and states.
	\param 'obj' (OUT) Created trace object.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT createTraceObject( TraceObject ** obj );

	/**
	\brief Function category: Callback registrations.
	\brief Register a callback function to signal a PnP Event of Baumer cameras.
	\param 'callBackOwner' (IN) Pointer to the user call back owner.
	\param 'PnPcallBack' (IN) Function pointer to a call back function.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT registerPnPCallback( void * callBackOwner, BGAPI_PNP_CALLBACK PnPcallBack );

	/**
	\brief Function category: Callback registrations.
	\brief Register a callback function to signal a bus reset of Baumer 1394 cameras.
	\param 'callBackOwner' (IN) Pointer to the user call back owner.
	\param 'BusResetCallBack' (IN) Function pointer to a call back function.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT registerBusResetCallback( void * callBackOwner, BGAPI_BUSRESET_CALLBACK BusResetCallBack );
};

/**
\brief Representatin of a BGAPI camera object. 
*/
class BGAPI_DECL Camera
{
private:
	/**
	\brief Function category: Constructor.
	\brief The constructor is a private function and can not directly called. To create an instance of this object use function System.createCamera.
	*/
	Camera();
    Camera( const Camera &Obj );
public:
	/**
	\brief Function category: Working with BGAPI objects.
	\brief This function opens a BGAPI camera object. Only after opening the features of BGAPI camera can be used.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT open();
	
	/**
	\brief Function category: BO Basic API.
	\brief This function returns the count of supported features of BGAPI system.
	\param 'count' (OUT) The count of BGAPI system features.	
	\retval 'BGAPI_RESULT_OK'	No error.
	*/  
	BGAPI_RESULT getFeatureCount( int* count );

	/**
	\brief Function category: BO Basic API.
	\brief This function executes a basic API feature.
	\param 'toset' (IN) The basic feature to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
    BGAPI_RESULT setFeature( Feature * toset );

	/**
	\brief Function category: BO Basic API.
	\brief This function refreshs the basic API feature state and returns the actual settings.
	\param 'Id' (IN) Identifier of basic API feature to refresh.
	\param 'cur' (OUT) The pointer to desired basic API feature.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT getFeature( BGAPI_FeatureId Id, Feature ** cur );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set a new image format. Use function getFormat to recive the available image formats.
	\param 'formatindex' (IN) Index of image format to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT setImageFormat( int formatindex );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns status informations of the current image format and a list with all available formats. The list returns only simple integers as format index. To get detailed description use function Camera.getFormatDescription and pass the index of desired format.
	\param 'pstate' (OUT) Current feature state.
	\param 'list' (OUT) Current format index with additional list information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getImageFormat( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the description of the image format, specified by 'formatindex'.  Please refer to enumeration BGAPI_CameraImageFormat, for detailed description.
	\param 'formatindex' (IN) Any valid format index.
	\param 'cformat' (OUT) Image format description.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getImageFormatDescription( int formatindex, BGAPIX_CameraImageFormat * cformat );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function gets a complete list of all features which are supported by the passed format index.
	\param 'formatindex' (IN) Any valid formatindex.
	\param 'flist' (OUT) List with supported features of desired camera format.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getFormatFeature( int formatindex, BGAPIX_TypeListFEATURE * flist );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set a new pixel format. Use function Camera.getPixelFormat and Camera.getPixelFormatDescription to get detailed descriptions.
	\param 'pixelformat' (IN) New pixel format.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setPixelFormat( int pixelformat );

	/**
	\brief Function category: BO Extended API.
	\brief This function gets a list with available pixel formats. The 'current' member of structure BGAPIX_TypeListINT is only valid, when passing the current 'formatindex'.
	\param 'formatindex' (IN) Any valid format index.
	\param 'pstate' (OUT) Current feature state.
	\param 'list' (OUT) Current pixel format with additional list information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getPixelFormat( int formatindex, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the description of the pixel format specified by 'formatindex' and 'pixelformat' Please refer to enumeration BGAPI_CameraPixelFormat, for detailed informations.
	\param 'formatindex' (IN) Any valid format index.
	\param 'pixelformat' (IN) Any valid pixelformat. Call function getPixelFormat to get all valid pixelformats of passed formatindex.
	\param 'pformat' (OUT) Description of desired camera pixel format.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getPixelFormatDescription( int formatindex, int pixelformat, BGAPIX_CameraPixelFormat * pformat );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to enable the trigger.
	\param 'enable' (IN) Switch camera trigger.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setTrigger( bool enable );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of trigger.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getTrigger( BGAPI_FeatureState * pstate );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to select a trigger source.
	\param 'trigger_source' (IN) Trigger Source. Please refer to enumeration BGAPI_TriggerSource. 
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setTriggerSource( BGAPI_TriggerSource trigger_source );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the active tigger source.
	\param 'pstate' (OUT) Current feature state.
	\param 'tSource' (OUT) Current and available trigger sources.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getTriggerSource( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tSource );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the trigger activation. Please refer to structure BGAPI_Activation to get detailed informations.
	\param 'activation' (IN) Trigger activation.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setTriggerActivation( BGAPI_Activation activation );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature, the active trigger activation and a list with available activation modes.
	\param 'pstate' (OUT) Current feature state.
	\param 'tActivation' (OUT) Current and available activation modes.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getTriggerActivation( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tActivation );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set a trigger delay.
	\param 'delay' (IN) Trigger delay to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setTriggerDelay( int delay );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and current used trigger delay.
	\param 'pstate' (OUT) Current feature state.
	\param 'tDelay' (OUT) Current trigger delay with additional range information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getTriggerDelay( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDelay );
    
	/**
	\brief Function category: BO Extended API.
	\brief Do software Trigger. Before using this function make sure that the trigger source software is activated. Please refer to function Camera.setTriggerSource.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT doTrigger( );    

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to enable the flash.
	\param 'enable' (IN) Enable/disable flash signal.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setFlash( bool enable );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of flash feature.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT getFlash( BGAPI_FeatureState * pstate );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the flash activation. Please refer to structure BGAPI_Activation to get detailed informations.
	\param 'activation' (IN) Set flash activation.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setFlashActivation( BGAPI_Activation activation );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of flash feature, the active flash activation and available activation modes for flash signal. The 'tActivation' list returns only simple integer values. This integers have to interpret as members of enumeration BGAPI_Activation.
	\param 'pstate' (OUT) Current feature state.
	\param 'tActivation' (OUT) Current flash activation with additional list information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getFlashActivation( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tActivation );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set a flash delay. This function also needs the trigger delay to calculate the flash duration. 
	\param 'flashdelay' (IN) Flash delay to set.
	\param 'triggerdelay' (IN) Trigger delay to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setFlashDelay( int flashdelay, int triggerdelay );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and current used flash delay.
	\param 'pstate' (OUT) Current feature state.
	\param 'tFlashDelay' (OUT) Current flash delay with additional range information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getFlashDelay( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tFlashDelay );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function activates the cooler.
	\param 'enable' (IN) Enable/disable cooler.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setCooler( bool enable );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns current informations about the state of cooler.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getCooler( BGAPI_FeatureState * pstate );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to enable the auto exposure feature. In addition you can specify a target brightness as gray levels and a ROI in which the AEC works.
	\param 'enable' (IN) Enable/disable auto exposure.
	\param 'brightpercent' (IN) Specify the target brightness.
	\param 'left' (IN) Left coordinate of auto exposure ROI.
	\param 'top' (IN) Top coordinate of auto exposure ROI.
	\param 'right' (IN) Right coordinate of auto exposure ROI.
	\param 'bottom' (IN) Bottom coordinate of auto exposure ROI.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setAEC( bool enable, int brightpercent, int left, int top, int right, int bottom );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns information about the auto exposure control.
	\param 'pstate' (OUT) Current feature state.
	\param 'brightpercent' (OUT) Current target brightness of auto exposure with additional range information.
	\param 'roi' (OUT) Current ROI of auto exposure.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getAEC( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * brightpercent, BGAPIX_TypeROI * roi );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the exposure time.
	\param 'exposure' (IN) Exposure time in µs.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setExposure( int exposure );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function return the state of this feature and the current exposure time with additional range informations.
	\param 'pstate' (OUT) Current feature state.
	\param 'sint' (OUT) Current exposure time with additional range informations.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getExposure( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * sint );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set a shutter mode. Please refer to enumeration BGAPI_Shutter to get detailed informations about the shutter modies.
	\param 'shutter' (IN) Shutter mode to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setShutter( BGAPI_Shutter shutter );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of the shutter mode and a list with supported shutter modies. The list returns only simple integer values. This integers have to interpret as members of enumeration BGAPI_Shutter.
	\param 'pstate' (OUT) Current feature state.
	\param 'list' (OUT) Current shutter mode with additional list information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getShutter( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the analog gain.
	\param 'gain' (IN) Gain to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setGain( float gain );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of analog gain and the current value with additional range informations.
	\param 'pstate' (OUT) Current feature state.
	\param 'sfloat' (OUT) Current gain value with additional range information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGain( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * sfloat);

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the digital gain.
	\param 'dgain' (IN) Digital gain to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setDigitalGain( int dgain );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of digital gain and the current value with additional range informations.
	\param 'pstate' (OUT) Current feature state.
	\param 'tDigitalGain' (OUT) Current gain value with additional range information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getDigitalGain( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDigitalGain );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the black level offset.
	\param 'offset' (IN) Black level offset to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setOffset(int offset );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of feature offset and the current value with additional range informations.
	\param 'pstate' (OUT) Current feature state.
	\param 'sint' (OUT) Current offset value with additional range information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getOffset( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * sint );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to activate the partial scan feature. 
	\param 'enable' (IN) Enable/disable partial scan.
	\param 'left' (IN) Left coordinate of partial scan.
	\param 'top' (IN) Top coordinate of partial scan.
	\param 'right' (IN) Right coordinate of partial scan.
	\param 'bottom' (IN) Bottom coordinate of partial scan.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setPartialScan( bool enable, int left, int top, int right, int bottom );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of partial scan feature and the current ROI.
	\param 'pstate' (OUT) Current feature state.
	\param 'roi' (OUT) Current partial scan ROI.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getPartialScan( BGAPI_FeatureState * pstate, BGAPIX_TypeROI * roi );

	/**
	\brief Function category: BO Extended API.
	\brief This function calculates the white balance factors in the specified rectangle.
	\param 'left' (IN) Left coordinate of white balance rectangle.
	\param 'top' (IN) Top coordinate of white balance rectangle.
	\param 'right' (IN) Right coordinate of white balance rectangle.
	\param 'bottom' (IN) Bottom coordinate of white balance rectangle.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT doWhiteBalance( int left, int top, int right, int bottom );

	/**
	\brief Function category: BO Extended API.
	\brief This function enables the white balance and uses the passed white balance factors.
	\param 'enable' (IN) Enable/disable white balance.
	\param 'red' (IN) Red factor of white balance.
	\param 'green' (IN) Green factor of white balance.
	\param 'blue' (IN) Blue factor of white balance.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setWhiteBalance( bool enable, float red, float green, float blue );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of feature white balance and the current white balance factors. To calculate these factors use function doWhiteBalance.
	\param 'pstate' (OUT) Current feature state.
	\param 'wbred' (OUT) Current red factor with additional range information.
	\param 'wbgreen' (OUT) Current green factor with additional range information.
	\param 'wbblue' (OUT) Current blue factor with additional range information.
	\param 'roi' (OUT) Current white balance ROI.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getWhiteBalance( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * wbred, BGAPIX_TypeRangeFLOAT * wbgreen, BGAPIX_TypeRangeFLOAT * wbblue, BGAPIX_TypeROI * roi );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to load the factory settings.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT doFactorySettings( );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of feature factory settings.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getFactorySettings( BGAPI_FeatureState * pstate );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to start/stop the camera.
	\param 'enable' (IN) Set true to start the camera, set false to stop it.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setStart( bool enable	);
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getStart( BGAPI_FeatureState * pstate );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to enable a test pattern, specified by 'mode'. Please refer to enumeration BGAPI_TestPattern to get detailed informations.
	\param 'enable' (IN) Enable/Disable test pattern.
	\param 'mode' (IN) Test pattern to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setTestPattern( bool enable, BGAPI_TestPattern mode );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function return the current state of test pattern feature and a list with supported test patterns. The list returns only simple integer values. This integers have to interpret as members of enumeration BGAPI_TestPattern.
	\param 'pstate' (OUT) Current feature state.
	\param 'list' (OUT) Current test pattern with additional list information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getTestPattern( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );

	/**
	\brief Function category: BO Extended API.
	\brief This function excecutes a asynchronous reset.
	\param 'delay' (IN) True to use a delay, false not.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT doAsyncReset(	bool delay );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of feature asynchronous reset.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getAsyncReset( BGAPI_FeatureState * pstate );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the readout tap of the camera sensor.
	\param 'readout' (IN) Set readout mode.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setReadout( BGAPI_Readout readout );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of readout feature and a list with supported readout taps. The list returns only simple integer values. This integers have to interpret as members of enumeration BGAPI_Readout.
	\param 'pstate' (OUT) Current feature state.
	\param 'list' (OUT) Current redout mode with additional list information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getReadout( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );

	/**
	\brief Function category: BO Extended API.
	\brief Deprecated, please use setDataAccessMode.
	\param 'mode' (IN) Data access mode to set.
	\param 'numberofbuffer' (IN) Number of buffers.
	\param 'useexternbuffer' (IN) Use this flag to set the buffer source (intern allocated buffer / extern allocated buffer). 
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setDataAccess( BGAPI_DataAccess mode, int numberofbuffer, bool useexternbuffer );

	/**
	\brief Function category: BO Extended API.
	\brief Deprecated, please use getDataAccessMode.
	\param 'pstate' (OUT) Current feature state.
	\param 'mode' (OUT) Current data access mode with a list of available data access modes.
	\param 'tnumberofbuffer' (OUT) Get the count of used buffers.
	\param 'tuseexternbuffer' (OUT) True, if you use your own buffers.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getDataAccess( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * mode, BGAPIX_TypeINT * tnumberofbuffer, BGAPIX_TypeBOOL * tuseexternbuffer );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to control the access to image data, by setting the buffer mode. Refer to enumeration BGAPI_DataAccess to get detailed informations about supported access modies.
	\param 'tMode' (IN) Data access mode to set.
	\param 'numberofbuffer' (IN) Number of buffers.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setDataAccessMode( BGAPI_DataAccessMode tMode, int numberofbuffer );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of data access feature and a list with supported data access modies. The list returns only simple integer values. This integers have to interpret as members of enumeration BGAPI_DataAccessMode.
	\param 'pstate' (OUT) Current feature state.
	\param 'list' (OUT) Current data access mode with additional list information.
	\param 'tnumberofbuffer' (OUT) Get the count of used buffers.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getDataAccessMode( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list, BGAPIX_TypeINT * tnumberofbuffer );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set a new start value of the software and hardware frame counter.
	\param 'swcounter' (IN) Set image software counter.
	\param 'hwcounter' (IN) Set camera generated counter.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setFrameCounter( int swcounter, int hwcounter );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of frame counter feature and the current counter value with additional range informations.
	\param 'pstate' (OUT) Current feature state.
	\param 'swrange' (OUT) Current software counter with additional range information.
	\param 'hwrange' (OUT) Current camera generated counter with additional range information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getFrameCounter( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * swrange, BGAPIX_TypeRangeINT * hwrange );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to reset the timestamp.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT resetTimeStamp();
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of time stamp feature, the current time stamp and the timestamp frequency.
	\param 'pstate' (OUT) Current feature state.
	\param 'timestamphigh' (OUT) Current timestamp. High part.
	\param 'timestamplow' (OUT) Current timestamp. Low part.
	\param 'frequencyhigh' (OUT) Current timestamp frequency. High part.
	\param 'frequencylow' (OUT) Current timestamp frequency. Low part.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getTimeStamp( BGAPI_FeatureState * pstate, BGAPIX_TypeINT * timestamphigh, BGAPIX_TypeINT * timestamplow, BGAPIX_TypeINT * frequencyhigh, BGAPIX_TypeINT * frequencylow );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function, to select a camera canal. For cameras mit more than one data channels (e.g. stereo camera).
	\param 'canal' (IN) Select camera canal. 
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setCanalSelect( int canal );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of canal feature and the current canal with additional list informations.
	\param 'pstate' (OUT) Current feature state.
	\param 'list' (OUT) Current camera canal with additional list information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getCanalSelect( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT *list );

	/**
	\brief Function category: BO Extended API.
	\brief Set a user defined gamma look up table.
	\param 'enable' (IN) Enable/disable gamma look up table.
	\param 'red' (IN) Red canal of look up table.
	\param 'green' (IN) Green canal of look up table.
	\param 'blue' (IN) Blue canal of look up table.
	\param 'size' (IN) Size of look up table. Maximum is 4096.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setLookupTableGamma( bool enable, int red[4096], int green[4096], int blue[4096], int size );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of look-up-table feature and the current look-up-table for the canals red, green and blue.
	\param 'pstate' (OUT) Current feature state.
	\param 'r' (OUT) Current red canal as large array.
	\param 'g' (OUT) Current green canal as large array.
	\param 'b' (OUT) Current blue canal as large array.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getLookupTableGamma( BGAPI_FeatureState * pstate, BGAPIX_TypeLargeArrayINT * r, BGAPIX_TypeLargeArrayINT * g, BGAPIX_TypeLargeArrayINT * b );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to send a camera command. This functinon returns the response.
	\param 'pcommand' (IN) Serial command string to camera.
	\param 'size' (IN) Size of serial command.
	\param 'response' (OUT) Camera response.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setCommand( char * pcommand, int size, BGAPIX_TypeSTRING * response );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of the command feature.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getCommand( BGAPI_FeatureState * pstate );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to enable the brightness correcion for binning modies.
	\param 'bSet' (IN) Enable/disable brightness correction.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setBrightnessCorrection( bool bSet );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of brightness correction feature.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getBrightnessCorrection( BGAPI_FeatureState * pstate );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns current device informations.
	\param 'pstate' (OUT) Current feature state.
	\param 'deviceinfo' (OUT) Device information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getDeviceInformation( BGAPI_FeatureState * pstate, BGAPIX_CameraInfo * deviceinfo ); 

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set user defined camera identifier/label.
	\param 'userid' (IN) User defined camera identifier/label.
	\param 'size' (IN) Size of user defined camera identifier/label.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setUserId( char * userid, int size );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of the UserId feature and the current used identifier/label.
	\param 'pstate' (OUT) Current feature state.
	\param 'tuserid' (OUT) Current used camera identifier/label.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT getUserId( BGAPI_FeatureState * pstate, BGAPIX_TypeSTRING * tuserid );

	/**
	\brief Function category: BO Extended API. Special Gige feature.
	\brief This function enables the heart beat timeout of Gige camera.
	\param 'benable' (IN) Flag to enable heartbeat timeout.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setGVSHeartBeatEnable( bool benable );

	/**
	\brief Function category: BO Extended API. Special Gige feature.
	\brief This function returns the current state of the heart beat timeout feature.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGVSHeartBeatEnable( BGAPI_FeatureState * pstate );

	/**
	\brief Function category: BO Extended API. Special Gige feature.
	\brief This function sets the heart beat timeout of Gige camera.
	\param 'htimeout' (IN) Timeout in ms.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setGVSHeartBeatTimeout( int htimeout );

	/**
	\brief Function category: BO Extended API. Special Gige feature.
	\brief This function returns the current state of this feature and the heart beat timeout of Gige camera with additional range informations.
	\param 'pstate' (OUT) Current feature state.
	\param 'timeinms' (OUT) Current used heart beat timeout. 
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGVSHeartBeatTimeout( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * timeinms );

	/**
	\brief Function category: BO Extended API. Special Gige feature.
	\brief Sets the GVS interpacket gap in ticks. The tick frequency can determined by function Camera.getTimeStamp.
	\param 'packetdelay' (IN) 
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setGVSPacketDelay( int packetdelay );

	/**
	\brief Function category: BO Extended API. Special Gige feature.
	\brief This function returns the state of this feature and the current GVS interpacket gap.
	\param 'pstate' (OUT) Current feature state.
	\param 'tPacketDelay' (OUT) 
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGVSPacketDelay( BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tPacketDelay );

	/**
	\brief Function category: BO Extended API. Special Gige feature.
	\brief This function enables or disables the persistent ip mode for a Gige camera. When enable this feature you have to pass the ip address, subnet mask and gateway. In other case these parametere are ignored. To apply these settings reconnect the power suply of camera.
	\param 'enable' (IN) Enable/Disable persistent ip mode.
	\param 'ip' (IN) Used IP address in persistent mode.
	\param 'subnetmask' (IN) Used subnet mask in persistent mode.
	\param 'gateway' (IN) Used gateway in persistent mode.
	\param 'size' (IN) Count of used elements in the arrays ip, subnetmask and gateway.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setGVSPersistentIP( bool enable, unsigned char ip[32], unsigned char subnetmask[32], unsigned char gateway[32], int size );

	/**
	\brief Function category: BO Extended API. Special Gige feature.
	\brief This function returns the state of this feature and the current persistent ip settings of a Gige camera.
	\param 'pstate' (OUT) Feature state.
	\param 'tenable' (OUT) Enable/Disable persistent ip mode.
	\param 'tip' (OUT) Used IP address in persistent mode.
	\param 'tsubnetmask' (OUT) Used subnet mask in persistent mode.
	\param 'tgateway' (OUT) Used gateway in persistent mode.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGVSPersistentIP( BGAPI_FeatureState * pstate, BGAPIX_TypeBOOL * tenable, BGAPIX_TypeArrayBYTE * tip, BGAPIX_TypeArrayBYTE * tsubnetmask, BGAPIX_TypeArrayBYTE * tgateway );

	/**
	\brief Function category: BO Extended API. Special Gige feature.
	\brief This function enables or disables the DHCP ip mode for a Gige camera. To apply these settings reconnect the power suply of GigE camera.
	\param 'enable' (IN) Enable/Disable DHCP ip mode.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setGVSDHCP( bool enable );

	/**
	\brief Function category: BO Extended API. Special Gige feature.
	\brief This function returns the state of this feature and the current DHCP ip settings of a Gige camera.
	\param 'pstate' (OUT) Feature state.
	\param 'tenable' (OUT) Enable/Disable DHCP ip mode.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGVSDHCP( BGAPI_FeatureState * pstate, BGAPIX_TypeBOOL * tenable );

	/**
	\brief Function category: BO Extended API. 
	\brief This function is supported for BGAPIX_DEVICETYPE_BAUMERCAM_DCAM cameras with format id 7 and BGAPIX_DEVICETYPE_BAUMERCAM_GIGE cameras to set the packet size.
	\param 'packetsize' (IN) The packet size to set.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setPacketSize( int packetsize );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the packet size of the current used image format. The function is supported for BGAPIX_DEVICETYPE_BAUMERCAM_DCAM cameras with DCAM format id 7 and BGAPIX_DEVICETYPE_BAUMERCAM_GIGE cameras.
	\param 'pstate' (OUT) Feature state.
	\param 'tpacketsize' (OUT) The current packet size.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getPacketSize( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tpacketsize );

	/**
	\brief Function category: BO Extended API.
	\brief This function sets the framerate in fps inside of a continuous range.
	\param 'framerate' (IN) The framerate to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setFramesPerSecondsContinuous( float framerate );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the framerate in fps.
	\param 'pstate' (OUT) Feature state.
	\param 'tframerate' (OUT) The framerate range.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getFramesPerSecondsContinuous( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tframerate );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function sets the framerate in fps for BGAPIX_DEVICETYPE_BAUMERCAM_DCAM cameras if DCAM standard formats.
	\param 'framerate' (IN) The framerate to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setFramesPerSeconds( float framerate );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the framerate in fps for standard DCAM formats. The function is only supported for BGAPIX_DEVICETYPE_BAUMERCAM_DCAM cameras with DCAM standard formats.
	\param 'formatindex' (IN) Any valid image format index.
	\param 'pstate' (OUT) Feature state.
	\param 'tframerate' (OUT) The framerate.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getFramesPerSeconds( int formatindex, BGAPI_FeatureState * pstate, BGAPIX_TypeListFLOAT * tframerate );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to reset the camera statistic.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT resetStatistic();
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of statistic feature and the current statistic values. The composition of statistic array depends of the camera type. See BGAPIX_CameraStatistic description to get the composition of statistic array. 
	\param 'pstate' (OUT) Current feature state.
	\param 'statistic' (OUT) Current statistic value.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getStatistic( BGAPI_FeatureState * pstate, BGAPIX_CameraStatistic * statistic );

	/**
	\brief Function category: BO Extended API.
	\brief Hardware supported sequnce of image aquisition. Use this feature to enable and start the sequencer. 
	\brief The sequencer functions are available, when the sequencer is enabled, in other case the state of the sequencer functions are locked.
	\brief To config the sequencer (sequencer mode, parameter set and sequence counters) you have to enable the sequencer but do not start it. After configuration call this function to start the sequencer.
	\param 'enable' (IN) Enable the sequencer to config the sequence parameter.
	\param 'start' (IN) Starts the sequencer after configuraion. Reset this parameter to stop sequencer.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setSequencer( bool enable, bool start );
	
	/**
	\brief Function category: BO Extended API.
	\brief Hardware supported sequnce of image aquisition. This function returns the state of sequencer feature and the current value of the 'start' parameter. 
	\brief The state of enable value is represent in BGAPI_FeatureState structure.
	\param 'pstate' (OUT) Current feature state.
	\param 'pstart' (OUT) Is the sequencer started.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getSequencer( BGAPI_FeatureState * pstate, BGAPIX_TypeBOOL * pstart );

	/**
	\brief Function category: BO Extended API.
	\brief Hardware supported sequnce of image aquisition. Use this function to config the run mode of sequencer. For run mode descriptions, please refer to structure BGAPI_SequencerRunMode.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'runmode' (IN) Set the run mode.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setSequencerRunMode( BGAPI_SequencerRunMode runmode );
	
	/**
	\brief Function category: BO Extended API.
	\brief Hardware supported sequnce of image aquisition. This function returns the state of sequencer run mode feature and the current used run mode.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'pstate' (OUT) Current feature state.
	\param 'prunmodelist' (OUT) The current used run mode and additional informations about supported runmodes.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getSequencerRunMode( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * prunmodelist );

	/**
	\brief Function category: BO Extended API.
	\brief Hardware supported sequnce of image aquisition. Use this function to make the sequencer ready to add a new parameter set. 
	\brief Call this function to switch to the next parameter set. To add values to the parameter set use the sequencer parameterset functions, like
	\brief setSequencerExposure and setSequencerGain.
	\brief Use the function getSequencerAddParameter to desire the maximum number of possible parameter sets, which you can be added by this function.
	\brief Please note, after the last parameter set this function may not been executed.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT doSequencerAddParameter();

	/**
	\brief Function category: BO Extended API.
	\brief Hardware supported sequnce of image aquisition. This function returns the state of sequencer add feature and the number of valid parameter sets.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'pstate' (OUT) Current feature state.
	\param 'pnumofparamsets' (OUT) The number of added parameter sets.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getSequencerAddParameter( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * pnumofparamsets );

	/**
	\brief Function category: BO Extended API.
	\brief Hardware supported sequnce of image aquisition. Use this function to reset the parameter sets of sequencer. 
	\brief Use the parameter set functions, like Camera.setSequencerExposure and Camera.setSequencerGain directly after this function to 
	\brief configure the first parameter set. To configure further parametersets use function Camera.dodequencerAddParameter.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT doSequencerReset( );

	/**
	\brief Function category: BO Extended API.
	\brief Hardware supported sequnce of image aquisition. This function returns the state of sequencer reset feature.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getSequencerReset( BGAPI_FeatureState * pstate );

	/**
	\brief Function category: BO Extended API.
	\brief Hardware supported sequnce of image aquisition. The sequencer implements three loops to define the image sequence. Use this function to config these three loops. Please refer to parameter description of the several counters.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'loopcounter' (IN) The loop counter specifies the repeats of a complete sequence. A complete sequence is one cycle through the number of parameter sets.
	\param 'repeatcounter' (IN)	The repeat counter specifies the repeats of one parameter set. This counter is the same for all defined parameter sets.
	\param 'triggercounter' (IN) The numer of frames per trigger. The frames per trigger value is independent of the other two counters. Therefore you can define several operation modes, e.g. one trigger per frame, one trigger per loop with a number of parameter set repeats (repeatcounter), one trigger with a numer of parameter sets. Please note that the number of frames per trigger must be a multiple of one of the following counters: loopcounter, repeatcounter, number of parameter sets.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setSequencerCounter( int loopcounter, int repeatcounter, int triggercounter );

	/**
	\brief Function category: BO Extended API.
	\brief Hardware supported sequnce of image aquisition. This function returns the state of sequencer counter feature and the current counter values. For a description of the sequencer counter please refer to function setSequencerCounter.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'pstate' (OUT) Current feature state.
	\param 'ploopcounter' (OUT) The current loop counter with additional range informations.
	\param 'prepeatcounter' (OUT) The current repeat counter with additional range informations.
	\param 'ptriggercounter' (OUT) The current number of frames per trigger with additional range information.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getSequencerCounter( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * ploopcounter, BGAPIX_TypeRangeINT * prepeatcounter, BGAPIX_TypeRangeINT * ptriggercounter );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to add the exposure time to the current sequencer parameter set.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'exposure' (IN) Exposure time in µs.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setSequencerExposure( int exposure );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the exposure time of current sequencer parameter set.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'pstate' (OUT) Current feature state.
	\param 'tExposure' (OUT) Current exposure time with additional range informations.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getSequencerExposure( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tExposure );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to add the analog gain to a sequencer parameter set.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'gain' (IN) Gain to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setSequencerGain( float gain );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and analog gain of current sequencer parameter set.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'pstate' (OUT) Current feature state.
	\param 'tGain' (OUT) Current gain value with additional range information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getSequencerGain( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tGain);

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to add the duty cycle of the output of PWM controller.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'pwm' (IN) Output of PWM controller.
	\param 'dutycycle' (IN) Duty cycle to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setSequencerPWMDutyCycle( BGAPI_PWM pwm, float dutycycle );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the duty cycle of the output of PWM controller for the current sequencer parameter set.
	\brief This function is available, when the sequencer is enabled. In other case the state of this functions is locked.
	\param 'pwm' (IN) Output of PWM controller.
	\param 'pstate' (OUT) Current feature state.
	\param 'tDutyCycle' (OUT) Current gain value with additional range information.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getSequencerPWMDutyCycle( BGAPI_PWM pwm, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tDutyCycle);

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the current readout time of camera in ms.
	\param 'pstate' (OUT) Current feature state.
	\param 'tReadoutTime' (OUT) The current readout time.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getReadoutTime( BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tReadoutTime );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature, the number of digital outputs and their current state.
	\param 'pstate' (OUT) Current feature state.
	\param 'tAvailable' (OUT) The number of digital outputs.
	\param 'tCurrent' (OUT) A bit field, which represents the current state of digital outputs. The number of valid LSB's is determind by the number of digital outputs.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getLines( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAvailable, BGAPIX_TypeINT * tCurrent );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature, the number of digital outputs.
	\param 'pstate' (OUT) Current feature state.
	\param 'tAvailable' (OUT) The number of digital outputs.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getAvailableLines( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAvailable );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and several informations about the specified line.
	\param 'line' (IN) The desired line.
	\param 'pstate' (OUT) Current feature state.
	\param 'polarity' (OUT) The polarity of specified line.
	\param 'tMode' (OUT) The current line mode and a list with available line modes (see BGAPI_LineMode).
	\param 'tOutputSource' (OUT) The current output source of desired line and a list with available output sources (see BGAPI_LineOutputSource). 
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getLineState( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPI_Polarity * polarity, BGAPIX_TypeListINT * tMode, BGAPIX_TypeListINT * tOutputSource );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the mode and the output source of a specified line.
	\param 'line' (IN) The desired line.
	\param 'mode' (IN) The desired line mode.
	\param 'outputsource' (IN) The desired output source.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setLineState( BGAPI_Line line, BGAPI_LineMode mode, BGAPI_LineOutputSource outputsource );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the line mode of the specified line.
	\param 'line' (IN) The desired line.
	\param 'pstate' (OUT) Current feature state.
	\param 'tMode' (OUT) The current line mode and a list with available line modes (see BGAPI_LineMode).
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getLineMode( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tMode );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the line mode of a specified line.
	\param 'line' (IN) The desired line.
	\param 'mode' (IN) The desired line mode.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setLineMode( BGAPI_Line line, BGAPI_LineMode mode );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the output source of the specified line.
	\param 'line' (IN) The desired line.
	\param 'pstate' (OUT) Current feature state.
	\param 'tOutputSource' (OUT) The current output source of desired line and a list with available output sources (see BGAPI_LineOutputSource). 
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getLineOutputSource( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tOutputSource );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the the output source of a specified line.
	\param 'line' (IN) The desired line.
	\param 'outputsource' (IN) The desired output source.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setLineOutputSource( BGAPI_Line line, BGAPI_LineOutputSource outputsource );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the low/hight time in micro seconds of debouncer. Attention: This function is only possible for input lines.
	\brief The low time defines after which time the level of input line is interpreted as low level. 
	\brief The hight time defines after which time the level of input line is interpreted as hight level. 
	\param 'line' (IN) The line to ask for state.
	\param 'pstate' (OUT) Current feature state.
	\param 'tLowTime' (OUT) Get the low time of debouncer.
	\param 'tHighTime' (OUT) Get the high time of debouncer.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getLineDebouncer( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT *tLowTime, BGAPIX_TypeRangeINT * tHighTime );

	/**
	\brief Function category: BO Extended API.
	\brief Sets the line deouncer time for low level of an input line.
	\param 'line' (IN) The desired line. Please note, it must be an input line.
	\param 'lowtime' (IN) Set the low time.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setLineDebouncerLow( BGAPI_Line line, int lowtime );

	/**
	\brief Function category: BO Extended API.
	\brief Sets the line deouncer time for high level of an input line.
	\param 'line' (IN) The desired line to set. Please note, it must be an input line.
	\param 'hightime' (IN) Set the high time.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setLineDebouncerHigh( BGAPI_Line line, int hightime );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the current polarity of all lines which are attached with the specified 'useroutput'.
	\param 'useroutput' (IN) The desired user output.
	\param 'pstate' (OUT) Current feature state.
	\param 'polarity' (OUT) Current polarity of specified user output.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getSingleOutput( BGAPI_UserOutput useroutput, BGAPI_FeatureState * pstate, BGAPI_Polarity * polarity );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the polarity for the specified user output.
	\param 'useroutput' (IN) The desired user output.
	\param 'polarity' (IN) Switch the polarity of user output.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setSingleOutput( BGAPI_UserOutput useroutput, BGAPI_Polarity polarity );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the polarity of all available user outputs.
	\param 'pstate' (OUT) Current feature state.
	\param 'tAvailable' (OUT) Available user outputs.
	\param 'tCurrent' (OUT) The current polarity of all user outputs. This value is represented by a one-hot code.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getParallelOutput( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAvailable, BGAPIX_TypeINT * tCurrent );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set multiple outputs at once.
	\param 'newstate' (IN) The state of outputs. This value is represented by a one-hot code.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setParallelOutput( int newstate );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and a mask of input lines.
	\param 'pstate' (OUT) Current feature state.
	\param 'tInputMask' (OUT) Mask of input lines.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getInputLineMask( BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tInputMask );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and a mask of output lines.
	\param 'pstate' (OUT) Current feature state.
	\param 'tOutputMask' (OUT) Mask of output lines.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getOutputLineMask( BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tOutputMask );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and an array with all supported BGAPI_MessagID's.
	\param 'pstate' (OUT) Current feature state.
	\param 'tAvailable' (OUT) A list with all supported messages.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getMessage( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAvailable );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to enbale/disable a message.
	\param 'messageid' (IN) The messageid to be set (see structure BGAPI_MessageId).
	\param 'messagestate' (IN) The message state to be set (see structure BGAPI_MessageState).
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setMessage( BGAPI_MessageId messageid, BGAPI_MessageState messagestate );
 
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the state of the specified message..
	\param 'messageid' (IN) The message id to ask for state.
	\param 'pstate' (OUT) Current feature state.
	\param 'tMessageState' (OUT) The state of specified the message id.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getMessageState( BGAPI_MessageId messageid, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tMessageState );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getDefectPixelCorrection( BGAPI_FeatureState * pstate );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to enable the defectpixel correction.
	\param 'enable' (IN) Enable the defectpixel correction.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setDefectPixelCorrection( bool enable );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the complete defect pixel map. 
	\brief A defect pixel is described by its x and y coordinates and a flag, which allows the user to enable/disable a pixel.
	\param 'pstate' (OUT) Current feature state.
	\param 'tCurrentIndex' (OUT) The current index in the arrays tX, tY and tEnabled.
	\param 'tX' (OUT) The x coordinates of the defect pixels.
	\param 'tXMin' (OUT) The minimum x coordinate.
	\param 'tXMax' (OUT) The maximum x coordinate.
	\param 'tY' (OUT) The y coordinates of the defect pixels.
	\param 'tYMin' (OUT) The minimum y coordinate.
	\param 'tYMax' (OUT) The maximum y coordinate.
	\param 'tEnabled' (OUT) The state of a pixel. If true, the pixel is used for pixel correction. Oterwise this pixel is ignored.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getDefectPixels( BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tCurrentIndex, BGAPIX_TypeLargeArrayINT * tX, BGAPIX_TypeINT *tXMin, BGAPIX_TypeINT *tXMax, BGAPIX_TypeLargeArrayINT * tY, BGAPIX_TypeINT *tYMin, BGAPIX_TypeINT *tYMax, BGAPIX_TypeLargeArrayBOOL * tEnabled );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to mark a pixel as defect. 
	\brief A defect pixel is described by its x and y coordinates and a flag, which allows the user to enable/disable a pixel.
	\param 'index' (IN) The index in the defect pixel map. An exsiting pixel at this index is overwritten.
	\param 'x' (IN) The new x coordinate of the defect pixel.
	\param 'y' (IN) The new y coordinate of the defect pixel.
	\param 'enable' (IN) The state of the defect pixel.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setDefectPixel( int index, int x, int y, bool enable );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the current used user set with a list of available user sets.
	\brief These list of available user sets have to be used only with functions Camera.setUserSet and Camera.doUserSetStore.
	\param 'pstate' (OUT) Current feature state.
	\param 'tUserSet' (OUT) Contains the current user set and a array with available user sets.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getUserSet( BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tUserSet );

	/**
	\brief Function category: BO Extended API.
	\brief This function sets a new user set as current.
	\param 'userset' (IN) The current user set.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setUserSet( BGAPI_UserSet userset );

	/**
	\brief Function category: BO Extended API.
	\brief Store the current camera settings as user set. 
	\param 'userset' (IN) The current user set.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT doUserSetStore( BGAPI_UserSet userset );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the default user set with a list of available user sets.
	\param 'pstate' (OUT) Current feature state.
	\param 'tDefaultUserSet' (OUT) Contains the current user set and a array with available user sets.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getUserSetDefault( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tDefaultUserSet );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function selects the default user set. The default user set is the start up setting of the camera.
	\param 'userset' (IN) The current user set.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setUserSetAsDefault( BGAPI_UserSet userset );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the current used camera gamma.
	\param 'pstate' (OUT) Current feature state.
	\param 'tGamma' (OUT) The current used gamma value in the camera.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getGamma( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tGamma );

	/**
	\brief Function category: BO Extended API.
	\brief This function selects the gamma value in the camera.
	\param 'gamma' (IN) The new gamma value.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setGamma( float gamma );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the current used look up table with a list of supported look up tables.
	\param 'pstate' (OUT) Current feature state.
	\param 'tLUT' (OUT) The current used look up table with list of supported look up tables.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getLookUpTable( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tLUT );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature for the specified look up table.
	\param 'lut' (IN) Desired look up table.
	\param 'pstate' (OUT) Current feature state of defined look up table.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getLookUpTableState( BGAPI_LookUpTable lut, BGAPI_FeatureState * pstate );

	/**
	\brief Function category: BO Extended API.
	\brief This function sets the enable state for the specified look up table.
	\param 'lut' (IN) Desired look up table.
	\param 'enable' (IN) New state.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setLookUpTableState( BGAPI_LookUpTable lut, bool enable );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the look up table values for specified look up table.
	\param 'lut' (IN) Desired look up table.
	\param 'pstate' (OUT) Current feature state.
	\param 'tCurrentIndex' (OUT) The last used index, which was used in function Camera.setLookUpTableValue with additional range informations. 
	\param 'tLUTValues' (OUT) The look up table values.
	\param 'tLUTMin' (OUT) The smallest possible value.
	\param 'tLUTMax' (OUT) The largest possible value.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getLookUpTableValues( BGAPI_LookUpTable lut, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tCurrentIndex, BGAPIX_TypeLargeArrayINT * tLUTValues, BGAPIX_TypeINT *tLUTMin, BGAPIX_TypeINT *tLUTMax );

	/**
	\brief Function category: BO Extended API.
	\brief This function set a look up table value for a specified look up table.
	\param 'lut' (IN) Desired look up table.
	\param 'index' (IN) The new value is written at this 'index'. The possible minimum and maximum values can be get with function Camera.getLookUpTableValues.
	\param 'lutvalue' (IN) The new value for the look up table.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setLookUpTableValue( BGAPI_LookUpTable lut, int index, int lutvalue );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the available temperature sensors.
	\param 'pstate' (OUT) Current feature state.
	\param 'tTemperatureSensors' (OUT) An array with available temperature sensors (see BGAPI_TemperatureSensor).
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getTemperatureSensors( BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tTemperatureSensors );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current temperature for the specified temeprature sensor.
	\param 'temperaturesensor' (IN) Desired temperature sensor.
	\param 'pstate' (OUT) Current feature state.
	\param 'tCurrentTemperature' (OUT) The current temperature of specified temperature sensor.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getTemperature( BGAPI_TemperatureSensor temperaturesensor, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tCurrentTemperature );


	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the critical temperature of specified temeprature sensor. If this temperature is reached,
	\brief the camera sends warning messages via message channel. Please refer to function Camera.setMessage.
	\param 'temperaturesensor' (IN) Desired temperature sensor.
	\param 'pstate' (OUT) Current feature state.
	\param 'tCriticalTemperature' (OUT) The critical temperature of specified temperature sensor.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getTemperatureCritical( BGAPI_TemperatureSensor temperaturesensor, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tCriticalTemperature );

	/**
	\brief Function category: BO Extended API.
	\brief This function sets the critical temperature value for the specified temperature sensor.
	\param 'temperaturesensor' (IN) Temperature sensor.
	\param 'criticaltemperature' (IN) The critical temperature.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setTemperatureCritical( BGAPI_TemperatureSensor temperaturesensor, float criticaltemperature );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the available outputs of the PWM (pulse width modulation) controller.
	\param 'pstate' (OUT) Current feature state.
	\param 'tPWM' (OUT) The current used PWM frequency.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getPWM( BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tPWM );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state of PWM (pulse width modulation) controller output.
	\param 'pwm' (IN) Desired output of PWM controler.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getPWMEnable( BGAPI_PWM pwm, BGAPI_FeatureState * pstate);
	
	/**
	\brief Function category: BO Extended API.
	\brief Set the current enable state of PWM (pulse width modulation) controler output.
	\param 'pwm' (IN) Desired output of PWM controler.
	\param 'enable' (IN) Enable the desired output PWM controler.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setPWMEnable( BGAPI_PWM pwm, bool enable );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the current used frequency for the specified PWM (pulse width modulation) controller output.
	\param 'pwm' (IN) Desired output of PWM controler.
	\param 'pstate' (OUT) Current feature state.
	\param 'tFrequency' (OUT) The current used frequency.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getPWMFrequency( BGAPI_PWM pwm, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tFrequency );

	/**
	\brief Function category: BO Extended API.
	\brief Sets the current frequency of specified PWM (pulse width modulation) controler output.
	\param 'pwm' (IN) Desired output of PWM controler.
	\param 'frequency' (IN) The current used frequency.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setPWMFrequency( BGAPI_PWM pwm, float frequency );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the current used duty cycle for the specified PWM (pulse width modulation) controller output.
	\param 'pwm' (IN) Desired output of PWM controler.
	\param 'pstate' (OUT) Current feature state.
	\param 'tDutyCycle' (OUT) The current used duty cycle.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getPWMDutyCycle( BGAPI_PWM pwm, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tDutyCycle );

	/**
	\brief Function category: BO Extended API.
	\brief Sets the current duty cylce for the specified PWM (pulse width modulation) controller output.
	\param 'pwm' (IN) Desired output of PWM controler.
	\param 'dutycycle' (IN) The current used duty cycle.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setPWMDutyCycle( BGAPI_PWM pwm, float dutycycle );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the current used over power threshold for the specified PWM (pulse width modulation) controller output.
	\param 'pwm' (IN) Desired output of PWM controler.
	\param 'pstate' (OUT) Current feature state.
	\param 'tOverPowerThreshold' (OUT) The current used over power threshold.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getPWMOverPowerThreshold( BGAPI_PWM pwm, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tOverPowerThreshold );

	/**
	\brief Function category: BO Extended API.
	\brief Sets the current over power threashold for the specified PWM (pulse width modulation) controller output.
	\param 'pwm' (IN) Desired output of PWM controler.
	\param 'overpowerthreshold' (IN) The current used over power threshold.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setPWMOverPowerThreshold( BGAPI_PWM pwm, float overpowerthreshold );

	/**
	\brief Function category: BO Extended API.
	\brief Deprecated. Please use function getDistanceAmplitudeThreshold instead.
	*/
	BGAPI_RESULT getDistanceAmplitudeThreshold( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * threshold );
	
	/**
	\brief Function category: BO Extended API.
	\brief Deprecated. Please use function setToFDistanceAmplitudeThreshold instead.
	*/
	BGAPI_RESULT setDistanceAmplitudeThreshold( int threshold );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the distance amplitude threashold. Special function for 3d cameras.
	\param 'pstate' (OUT) Current feature state.
	\param 'threshold' (OUT) The current used distance amplitude threshold.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getToFDistanceAmplitudeThreshold( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * threshold );
	
	/**
	\brief Function category: BO Extended API.
	\brief Sets distance amplitude threashold. Special function for 3d cameras.
	\param 'threshold' (IN) The desired distance amplitude threshold.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setToFDistanceAmplitudeThreshold( int threshold );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the ToF modulation frequency. Special function for 3d cameras.
	\brief The function also returns a list with supported frequencies (see enumeration BGAPI_ToFModulationFrequency).
	\param 'pstate' (OUT) Current feature state.
	\param 'tFrequency' (OUT) The current used modulation frequency.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getToFModulationFrequency( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tFrequency );
	
	/**
	\brief Function category: BO Extended API.
	\brief Sets the ToF modulation frequency. Special function for 3d cameras.
	\param 'frequency' (IN) The desired modulation frequency in MHz.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setToFModulationFrequency( BGAPI_ToFModulationFrequency frequency );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the current line potential.
	\param 'pstate' (OUT) Current feature state.
	\param 'list' (OUT) The current used line potential and all available line potentials.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getLinePotential( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );
	
	/**
	\brief Function category: BO Extended API.
	\brief Set the line potential.
	\param 'linePotential' (IN) The desired line potential.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setLinePotential( BGAPI_LinePotential linePotential );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current used data length. It depends on the current camera settings, like pixel format.
	\param 'pstate' (OUT) Current feature state.
	\param 'tDataLength' (OUT) The current used data length.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getDataLength( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDataLength );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current size of the image.
	\param 'pstate' (OUT) Current feature state.
	\param 'tDataLength' (OUT) The current width.
	\param 'tDataLength' (OUT) The current hight.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getImageResolution( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tWidth, BGAPIX_TypeRangeINT * tHeight );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to enable the multicast mechanism for the image channel.
	\param 'benable' (IN) Enable the multicast.
	\param 'multicastip' (IN) Set the multicast ip address into the camera. If the first parameter is false, the multicast ip will ignored.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setMulticast( bool benable, unsigned char multicastip[32] );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current used multicast address and the current state. Use the enabled state in 'pstate' to check, if the multicast is enabled.
	\param 'pstate' (OUT) Current feature state.
	\param 'tMulticastIP' (OUT) The current used multicast ip address.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getMulticast( BGAPI_FeatureState * pstate, BGAPIX_TypeArrayBYTE * tMulticastIP );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to enable the multicast mechanism for the message channel.
	\param 'benable' (IN) Enable the multicast.
	\param 'messagemulticastip' (IN) Set the multicast ip address into the camera. If the first parameter is false, the multicast ip will ignored.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setMulticastMessage( bool benable, unsigned char multicastmessageip[32] );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current used multicast address and the current state. Use the enabled state in 'pstate' to check, if the multicast is enabled.
	\param 'pstate' (OUT) Current feature state.
	\param 'tMulticastMessageIP' (OUT) The current used multicast ip address.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getMulticastMessage( BGAPI_FeatureState * pstate, BGAPIX_TypeArrayBYTE * tMulticastMessageIP );

	/**
	\brief Function category: BO Extended API.
	\brief Check, if the configuration mode of camera/trigger device is enabled.
	\param 'pstate' (OUT) Current feature state.
	\param 'tEnable' (OUT) Current state of enable switch.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getConfigMode( BGAPI_FeatureState * pstate, BGAPIX_TypeBOOL * tEnable );

	/**
	\brief Function category: BO Extended API.
	\brief Enable the configuration mode of camera/trigger device. This allows to use all line configuration functions of a trigger device.
	\param 'bEnable' (IN) Enable switch.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setConfigMode( bool bEnable );	
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature, the current line activation and a list with available activation modes of a special line.
	\brief This feature is only available for input lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tActivation' (OUT) Current and available activation modes.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT getLineActivation( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tActivation );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the line activation of a desired line. Please refer to structure BGAPI_Activation to get detailed informations.
	\brief This feature is only available for input lines.	
	\param 'line' (IN) Line selector.
	\param 'tActivation' (IN) Line activation mode.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLineActivation( BGAPI_Line line, BGAPI_Activation tActivation );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current line delay in ticks for a desired line.
	\brief This feature is only available for input lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tDelay' (OUT) Current delay time in ticks.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLineDelay( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDelay );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the line delay of a desired line.
	\brief This feature is only available for input lines.	
	\param 'line' (IN) Line selector.
	\param 'tActivation' (IN) Delay time in ticks.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLineDelay( BGAPI_Line line, unsigned int delay );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current selected clock line, which is used as timebase.
	\brief The clock lines, which can be used for timebase are delivered by tTimeBaseLines. Please also refer to our technical documentation
	\brief of trigger device to get detailed information about the clock lines. 
	\brief This feature is only available for input lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tTimeBaseLines' (OUT) Current used clock line and a list with available clock lines.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLineDelayTimeBase( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tTimeBaseLines );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the line delay of a desired line.
	\brief This feature is only available for input lines.	
	\param 'line' (IN) Line selector.
	\param 'timebaseline' (IN) Clock line, which should be used as timebase. This line must be an input line.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLineDelayTimeBase( BGAPI_Line line, BGAPI_Line timebaseline );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current used clock rate divider for a desired line
	\brief This feature is only available for input lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tClockRateDivider' (OUT) Current used clock rate divider.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLineDelayClockRateDivider( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tClockRateDivider );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the clock rate divider for the line delay of a desired line.
	\brief This feature is only available for input lines.	
	\param 'line' (IN) Line selector.
	\param 'clockratedivider' (IN) Clock rate divider.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLineDelayClockRateDivider( BGAPI_Line line, int clockratedivider );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and a range of valid action commands.
	\brief Use this feature, when programming an action command receiver.
	\param 'pstate' (OUT) Current feature state.
	\param 'tActionCommands' (OUT) Gets a range of valid action commands. The action commands were represent by integer values.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getActionCommands( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tActionCommands );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and informations about the group key and group mask for a desired action command.
	\brief Use this feature, when programming an action command receiver.
	\param 'actioncommand' (IN) Action command selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tGroupKey' (OUT) The current group key.
	\param 'tGroupMask' (OUT) The current group mask.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getActionCommand( unsigned int actioncommand, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tGroupKey, BGAPIX_TypeINT * tGroupMask );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to configure an action command with groupkey and groupmask for a desired action command.
	\brief Use this feature, when programming an action command receiver.
	\param 'actioncommand' (IN) Action command selector.
	\param 'groupkey' (IN) Group key.
	\param 'groupmask' (IN) For every group key a 32 bit group mask can specified.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setActionCommand( unsigned int actioncommand, unsigned int groupkey, unsigned int groupmask );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to configure the device key for action command. The device key is write only.
	\brief Use this feature, when programming an action command receiver.
	\param 'devicekey' (IN) Device key.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setActionDeviceKey( unsigned int devicekey );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and a range of valid action commands.
	\brief Use this feature, when programming an action command sender.
	\param 'pstate' (OUT) Current feature state.
	\param 'tActionCommands' (OUT) Gets a range of valid action commands. The action commands were represent by integer values.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getSenderActionCommands( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tActionCommands );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and informations about the group key and group mask for a desired action command.
	\brief Use this feature, when programming an action command sender.
	\param 'actioncommand' (IN) Action command selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tGroupKey' (OUT) The current group key.
	\param 'tGroupMask' (OUT) The current group mask.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getSenderActionCommand( unsigned int actioncommand, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tGroupKey, BGAPIX_TypeINT * tGroupMask );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to configure an action command with devicekey, groupkey and groupmask for a desired action command.
	\brief Use this feature, when programming an action command sender.
	\param 'actioncommand' (IN) Action command selector.
	\param 'devicekey' (IN) Device key.
	\param 'groupkey' (IN) Group key.
	\param 'groupmask' (IN) For every group key a 32 bit group mask can specified.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setSenderActionCommand( unsigned int actioncommand, unsigned int devicekey, unsigned int groupkey, unsigned int groupmask );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the number of registered action command responders for a desired action command.
	\brief Use this feature, when programming an action command sender.
	\param 'actioncommand' (IN) Action command selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tNumberOf' (OUT) The number of registered action command responders.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getSenderNumberOfResponder( unsigned int actioncommand, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tNumberOf );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and a registered action command responder specified by an index.
	\brief Use this feature, when programming an action command sender.
	\param 'actioncommand' (IN) Action command selector.
	\param 'index' (IN) Index into the responder list.
	\param 'pstate' (OUT) Current feature state.
	\param 'tResponder' (OUT) The registered responder at desired index.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getSenderResponder( unsigned int actioncommand, int index, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayBYTE * tResponder );
	
	/**
	\brief Function category: BO Extended API.
	\brief If an action command with acknowledge is send, the action command sender must know all devices, which should respond the action command.
	\brief These devices are stored in a responder list. Use this function to add a responder to the responder list.
	\brief Use this feature, when programming an action command sender.
	\param 'actioncommand' (IN) Action command selector.
	\param 'responder' (IN) The responding device. For GigE devices this value needs IPV4 address.
	\retval 'BGAPI_RESULT_OK' No error.
	*/	
	BGAPI_RESULT doSenderResponderListAdd( unsigned int actioncommand, unsigned char responder[32] );

	/**
	\brief Function category: BO Extended API.
	\brief Reset the action command responder list for a desired action command.
	\brief Use this feature, when programming an action command sender.
	\param 'actioncommand' Action commmand selector.
	\retval 'BGAPI_RESULT_OK' No error.
	*/	
	BGAPI_RESULT doSenderResponderListReset( unsigned int actioncommand );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the state of this feature and the current used abort conditions for a desired action command.
	\brief Use this feature, when programming an action command sender.
	\param 'actioncommand' (IN) Action command selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tTimeout' (OUT) Timeout for an action command.
	\param 'tRetry' (OUT) Action command retries.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT getSenderAbortConditions( unsigned int actioncommand, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tTimeout, BGAPIX_TypeINT * tRetry );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to configure the abort conditions of a desired action command.
	\brief Use this feature, when programming an action command sender.
	\param 'actioncommand' (IN) Action command selector.
	\param 'timeout' (IN) The timeout in ticks.
	\param 'retry' (IN) The retries of an action command.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setSenderAbortConditions( unsigned int actioncommand, int timeout, int retry ); 

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current pulse delay in ticks for a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tPulseDelay' (OUT) Current pulse delay time in ticks.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLinePulseDelay( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tPulseDelay );	
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the pulse delay of a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'pulsedelay' (IN) Delay time in ticks.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLinePulseDelay( BGAPI_Line line, unsigned int pulsedelay );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current selected clock line, which is used as timebase for the pulse delay.
	\brief The clock lines, which can be used for timebase are delivered by tTimeBaseLines. Please also refer to our technical documentation
	\brief of trigger device to get detailed information about the clock lines. 
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tTimeBaseLines' (OUT) Current used clock line and a list with available clock lines.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLinePulseDelayTimeBase( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tTimeBaseLines );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the pulse delay of a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'timebaseline' (IN) Clock line, which should be used as timebase. This line must be an input line.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLinePulseDelayTimeBase( BGAPI_Line line, BGAPI_Line timebaseline );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current used clock rate divider for a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tClockRateDivider' (OUT) Current used clock rate divider.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLinePulseDelayClockRateDivider( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tClockRateDivider );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the clock rate divider for the pulse delay of a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'clockratedivider' (IN) Clock rate divider.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLinePulseDelayClockRateDivider( BGAPI_Line line, int clockratedivider );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current pulse width in ticks for a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tPulseWidth' (OUT) Current pulse width in ticks.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLinePulseWidth( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tPulseWidth );	
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the pulse width in ticks of a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'pulsewidth' (IN) Pulse width in ticks.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLinePulseWidth( BGAPI_Line line, unsigned int pulsewidth );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current selected clock line, which is used as timebase for the pulse width.
	\brief The clock lines, which can be used for timebase are delivered by tTimeBaseLines. Please also refer to our technical documentation
	\brief of trigger device to get detailed information about the clock lines. 
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tTimeBaseLines' (OUT) Current used clock line and a list with available clock lines.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLinePulseWidthTimeBase( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tTimeBaseLines );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the pulse width of a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'timebaseline' (IN) Clock line, which should be used as timebase. This line must be an input line.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLinePulseWidthTimeBase( BGAPI_Line line, BGAPI_Line timebaseline );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current used clock rate divider for the pulse width of a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tClockRateDivider' (OUT) Current used clock rate divider.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLinePulseWidthClockRateDivider( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tClockRateDivider );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the clock rate divider for the pulse width of a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'clockratedivider' (IN) Clock rate divider.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLinePulseWidthClockRateDivider( BGAPI_Line line, int clockratedivider );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature, the current selected gate mode and a list with available gate modes (see BGAPI_LineGate strucure) of a special line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tGateMode' (OUT) Current used gate mode.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLineGate( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tGateMode );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the gate mode for a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'gatemode' (IN) New gate mode.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLineGate( BGAPI_Line line, BGAPI_LineGate gatemode );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current used retrigger time in ticks for a desired line.
	\brief This feature is only available for output lines and by active trigger source BGAPI_TRIGGERSOURCE_RETRIGGER.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tReTriggerTime' (OUT) Current retrigger time in ticks.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLineReTriggerTime( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tReTriggerTime );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the retrigger in ticks of a desired line.
	\brief This feature is only available for output lines and by active trigger source BGAPI_TRIGGERSOURCE_RETRIGGER.
	\param 'line' (IN) Line selector.
	\param 'retriggertime' (IN) Retrigger time in ticks.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLineReTriggerTime( BGAPI_Line line, unsigned int retriggertime );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current selected clock line, which is used as timebase for the retrigger time.
	\brief The clock lines, which can be used for timebase are delivered by tTimeBaseLines. Please also refer to our technical documentation
	\brief of trigger device to get detailed information about the clock lines. 
	\brief This feature is only available for output lines and by active trigger source BGAPI_TRIGGERSOURCE_RETRIGGER.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tTimeBaseLines' (OUT) Current used clock line and a list with available clock lines.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLineReTriggerTimeBase( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tTimeBaseLines );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the retrigger timebase of a desired line.
	\brief This feature is only available for output lines and by active trigger source BGAPI_TRIGGERSOURCE_RETRIGGER.
	\param 'line' (IN) Line selector.
	\param 'timebaseline' (IN) Clock line, which should be used as timebase. This line must be an input line.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLineReTriggerTimeBase( BGAPI_Line line, BGAPI_Line timebaseline );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and the current used clock rate divider for the retrigger time of a desired line.
	\brief This feature is only available for output lines and by active trigger source BGAPI_TRIGGERSOURCE_RETRIGGER.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tClockRateDivider' (OUT) Current used clock rate divider.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLineReTriggerClockRateDivider( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tClockRateDivider );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the clock rate divider for the retrigger time of a desired line.
	\brief This feature is only available for output lines and by active trigger source BGAPI_TRIGGERSOURCE_RETRIGGER.
	\param 'line' (IN) Line selector.
	\param 'clockratedivider' (IN) Clock rate divider.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLineReTriggerClockRateDivider( BGAPI_Line line, int clockratedivider );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature and shows if the desired line is inverted.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'pstate' (OUT) Current feature state.
	\param 'tInverted' (OUT) Current invert state.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getLineInverter( BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeBOOL * tInverted );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to invert a desired line.
	\brief This feature is only available for output lines.
	\param 'line' (IN) Line selector.
	\param 'binvert' (IN) Switch to invert line.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setLineInverter( BGAPI_Line line, bool binvert );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the current state of this feature, the current used flip type and a list with available image flip types.
	\param 'pstate' (OUT) Current feature state.
	\param 'tFlipType' (OUT) Current invert state.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT getFlipType( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tFlipType );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to flip the image. The image is flipping directly by hardware.
	\param 'fliptype' (IN) Switch to invert line.
	\retval 'BGAPI_RESULT_OK' No error.
	*/
	BGAPI_RESULT setFlipType( BGAPI_FlipType fliptype );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to selecte the processing unit for automatic exposure control.
	\param 'processingunit' (IN) The proccessing unit.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setExposureAuto( BGAPI_ProcessingUnit processingunit );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns information about the curent and available processing units for auto exposure control.
	\param 'pstate' (OUT) Current feature state.
	\param 'tProcessingUnit' (OUT) Current processing unit and a list with available units.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getExposureAuto( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tProcessingUnit );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to selecte the automatic mode for an automatic operation. 
	\param 'automode' (IN) It can be one of the following modes: off, once or continuous mode.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setExposureAutoMode( BGAPI_AutomaticMode automode );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns information about the curent and available automatic modes for auto exposure control.
	\param 'pstate' (OUT) Current feature state.
	\param 'tAutoMode' (OUT) Current automatic mode and a list with available modes.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getExposureAutoMode( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAutoMode );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the target brightness for automatic exposure control.
	\param 'brightinpercent' (IN) The target brightness in percent.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setExposureAutoBrightness( float brightinpercent );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns information about the curent target brightness value for auto exposure control.
	\param 'pstate' (OUT) Current feature state.
	\param 'tBrightInPercent' (OUT) Current target brightness value.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getExposureAutoBrightness( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tBrightInPercent );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set a ROI in which the automatic exposure control works.
	\param 'left' (IN) The left coordinate of ROI.
	\param 'top' (IN) The top coordinate of ROI.
	\param 'right' (IN) The right coordinate of ROI.
	\param 'bottom' (IN) The bottom coordinate of ROI.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setExposureAutoROI( int left, int top, int right, int bottom );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent used ROI for auto exposure control.
	\param 'pstate' (OUT) Current feature state.
	\param 'tROI' (OUT) Current ROI.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getExposureAutoROI( BGAPI_FeatureState * pstate, BGAPIX_TypeROI *tROI );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set minimum limit of automatic exposure algorithm.
	\param 'minexposure' (IN) The minimum exposure value.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setExposureAutoMinimum( int minexposure );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent used minimum limit of auto exposure algorithm.
	\param 'pstate' (OUT) Current feature state.
	\param 'tMinExposure' (OUT) Current used minimum limit.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getExposureAutoMinimum( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tMinExposure );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set maximum limit of automatic exposure algorithm.
	\param 'maxexposure' (IN) The maximum exposure value.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setExposureAutoMaximum( int maxexposure );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent used maximum limit which is used by the auto exposure algorithm.
	\param 'pstate' (OUT) Current feature state.
	\param 'tMaxExposure' (OUT) Current used maximum limit.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getExposureAutoMaximum( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tMaxExposure );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent used 'L' offset for Lab color space.
	\param 'pstate' (OUT) Current feature state.
	\param 'tOffsetL' (OUT) Current used 'L' offset.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getOffsetL( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tOffsetL );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the offset in 'L' direction for Lab color space.
	\param 'offsetL' (IN) The offset in 'L' direction for Lab color space.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setOffsetL(  int offsetL );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent used 'a' offset for Lab color space.
	\param 'pstate' (OUT) Current feature state.
	\param 'tOffseta' (OUT) Current used 'a' offset.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getOffseta( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tOffseta );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the offset in 'a' direction for Lab color space.
	\param 'offseta' (IN) The offset in 'a' direction for Lab color space.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setOffseta(  int offseta );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent used 'b' offset for Lab color space.
	\param 'pstate' (OUT) Current feature state.
	\param 'tOffsetb' (OUT) Current used 'b' offset.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getOffsetb( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tOffsetb );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the offset in 'b' direction for Lab color space.
	\param 'offsetb' (IN) The offset in 'b' direction for Lab color space.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setOffsetb(  int offsetb );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent used gain factor for 'L' direction of Lab color space.
	\param 'pstate' (OUT) Current feature state.
	\param 'tGainL' (OUT) Current used gain factor for 'L' direction.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGainL( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tGainL );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the gain factor in 'L' direction for Lab color space.
	\param 'gainL' (IN) The gain factor in 'L' direction for Lab color space.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setGainL(  float gainL );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent used gain factor for 'a' direction of Lab color space.
	\param 'pstate' (OUT) Current feature state.
	\param 'tGaina' (OUT) Current used gain factor for 'a' direction.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGaina( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tGaina );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the gain factor in 'a' direction for Lab color space.
	\param 'gaina' (IN) The gain factor in 'a' direction for Lab color space.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setGaina(  float gaina );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent used gain factor for 'b' direction of Lab color space.
	\param 'pstate' (OUT) Current feature state.
	\param 'tGainb' (OUT) Current used gain factor for 'b' direction.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGainb( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tGainb );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the gain factor in 'b' direction for Lab color space.
	\param 'gainb' (IN) The gain factor in 'b' direction for Lab color space.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setGainb(  float gainb );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the transmission delay of camera.
	\param 'delay' (IN) The transmission delay in ticks.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setTransmissionDelay( int delay );
	
	/**
	\brief Function category: BO Image API.
	\brief This function return the state of this feature and the current used transmission delay with additional range informations.
	\param 'pstate' (OUT) Current feature state.
	\param 'tDelay' (IN) Current used transmission delay in ticks.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getTransmissionDelay( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDelay );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the scaling factor in x direction.
	\param 'xfactor' (IN) Scaling factor in x direction.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setScalingFactorX( float xfactor );
	
	/**
	\brief Function category: BO Image API.
	\brief This function return the state of this feature and the current used scaling factor in x direction.
	\param 'pstate' (OUT) Current feature state.
	\param 'tFactorX' (IN) Current used scaling factor in x direction.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getScalingFactorX( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tFactorX );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the scaling factor in y direction.
	\param 'yfactor' (IN) Scaling factor in y direction.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setScalingFactorY( float yfactor );
	
	/**
	\brief Function category: BO Image API.
	\brief This function return the state of this feature and the current used scaling factor in y direction.
	\param 'pstate' (OUT) Current feature state.
	\param 'tFactorX' (IN) Current used scaling factor in y direction.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getScalingFactorY( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tFactorY );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the available timers of camera. A timer can be used to control the signal of an output line.
	\param 'pstate' (OUT) Current feature state.
	\param 'tTimer' (OUT) The supported timers.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getTimers( BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tTimer );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the activation duration of the selected timer.
	\param 'timer' (IN) The duration is queried for this timer.
	\param 'pstate' (OUT) Current feature state.
	\param 'tDuration' (OUT) The current used duration in mircoseconds.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getTimerDuration( BGAPI_Timer timer, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDuration );
	
	/**
	\brief Function category: BO Extended API.
	\brief Sets the duration of specified timer.
	\param 'timer' (IN) The duration is set for this timer.
	\param 'duration' (IN) The duration in mircoseconds.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setTimerDuration( BGAPI_Timer timer, int duration );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the delay until to the activation of output signal.
	\param 'timer' (IN) The delay is queried for this timer.
	\param 'pstate' (OUT) Current feature state.
	\param 'tDelay' (OUT) The current used delay in mircoseconds.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getTimerDelay( BGAPI_Timer timer, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDelay );
	
	/**
	\brief Function category: BO Extended API.
	\brief Sets the delay until to the activation of output signal.
	\param 'timer' (IN) The delay is set for this timer.
	\param 'delay' (IN) The delay in mircoseconds.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setTimerDelay( BGAPI_Timer timer, int delay );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the current used trigger source for the specified timer.
	\param 'timer' (IN) The trigger source is queried for this timer.
	\param 'pstate' (OUT) Current feature state.
	\param 'tSource' (OUT) The current used trigger source.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getTimerTriggerSource( BGAPI_Timer timer, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tSource );
	
	/**
	\brief Function category: BO Extended API.
	\brief Sets the trigger source for the specified timer.
	\param 'timer' (IN) The trigger source is set for this timer.
	\param 'source' (IN) The new trigger source.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setTimerTriggerSource( BGAPI_Timer timer, BGAPI_TriggerSource source );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the current used trigger activation for the specified timer.
	\param 'timer' (IN) The trigger activation is queried for this timer.
	\param 'pstate' (OUT) Current feature state.
	\param 'tActivation' (OUT) The current used trigger activation.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getTimerTriggerActivation( BGAPI_Timer timer, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tActivation );
	
	/**
	\brief Function category: BO Extended API.
	\brief Sets the trigger activation for the specified timer.
	\param 'timer' (IN) The trigger activation is set for this timer.
	\param 'activation' (IN) The new trigger activation.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setTimerTriggerActivation( BGAPI_Timer timer, BGAPI_Activation activation );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to selecte the automatic mode for an automatic operation. 
	\param 'automode' (IN) It can be one of the following modes: off, once or continuous mode.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setGainAutoMode( BGAPI_AutomaticMode automode );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns information about the curent and available automatic modes for auto gain control.
	\param 'pstate' (OUT) Current feature state.
	\param 'tAutoMode' (OUT) Current automatic mode and a list with available modes.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGainAutoMode( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAutoMode );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set maximum limit of automatic gain algorithm.
	\param 'maxgain' (IN) The maximum gain value.
	\retval 'BGAPI_RESULT_OK' No error.
	*/	
	BGAPI_RESULT setGainAutoMaximum( float maxgain );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent used maximum limit which is used by the auto gain algorithm.
	\param 'pstate' (OUT) Current feature state.
	\param 'tMaxGain' (OUT) Current used maximum limit.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getGainAutoMaximum( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tMaxGain );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the available counters of camera. A counter can be used to monitor the change of edges.
	\param 'pstate' (OUT) Current feature state.
	\param 'tCounter' (OUT) The supported counters.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getCounters( BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tCounter );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the current counter value.
	\param 'counter' (IN) The value is queried for this counter.
	\param 'pstate' (OUT) Current feature state.
	\param 'tValue' (OUT) The current counter value.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getCounter( BGAPI_Counter counter, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tValue );
	
	/**
	\brief Function category: BO Extended API.
	\brief Sets the value of specified counter.
	\param 'counter' (IN) The value is set for this counter.
	\param 'value' (OUT) The new counter value.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT setCounter( BGAPI_Counter counter, int value );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the current used trigger source for the specified counter.
	\param 'counter' (IN) The trigger source is queried for this counter.
	\param 'pstate' (OUT) Current feature state.
	\param 'tSource' (OUT) The current used trigger source.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getCounterTriggerSource( BGAPI_Counter counter, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tSource );
	
	/**
	\brief Function category: BO Extended API.
	\brief Sets the trigger source for the specified counter.
	\param 'counter' (IN) The trigger source is set for this counter.
	\param 'source' (IN) The new trigger source.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setCounterTriggerSource( BGAPI_Counter counter, BGAPI_TriggerSource source );
	
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the feature state and the current used trigger activation for the specified counter.
	\param 'counter' (IN) The trigger activation is queried for this counter.
	\param 'pstate' (OUT) Current feature state.
	\param 'tActivation' (OUT) The current used trigger activation.
	\retval 'BGAPI_RESULT_OK'
	*/	
	BGAPI_RESULT getCounterTriggerActivation( BGAPI_Counter counter, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tActivation );
	
	/**
	\brief Function category: BO Extended API.
	\brief Sets the trigger activation for the specified counter.
	\param 'timer' (IN) The trigger activation is set for this counter.
	\param 'activation' (IN) The new trigger activation.
	\retval 'BGAPI_RESULT_OK'
	*/
	BGAPI_RESULT setCounterTriggerActivation( BGAPI_Counter counter, BGAPI_Activation activation );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns information about the curent and available processing units for high dynamic range feature (HDR).
	\param 'pstate' (OUT) Current feature state.
	\param 'tProcessingUnit' (OUT) Current processing unit and a list with available units.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getHDRProcessingUnit( BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tProcessingUnit );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to selecte the processing unit for high dynamic range feature (HDR).
	\param 'processingunit' (IN) The proccessing unit.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setHDRProcessingUnit( BGAPI_ProcessingUnit processingunit );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent state of HDR feature. It depends on the selected HDR processing unit.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getHDREnable( BGAPI_FeatureState * pstate );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to enable the HDR feature for the selected processing unit.
	\param 'benable' (IN) Switch to enable the HDR feature.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setHDREnable( bool benable );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent state of this feature and the number of images, which are used by HDR feature.
	\param 'pstate' (OUT) Current feature state.
	\param 'tNumberOfImages' (OUT) The current used number of images and additional range informations.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getHDRNumberOfImages( BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tNumberOfImages );
	
	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set the number of images to be used by HDR feature.
	\param 'numberofimages' (IN) The number of images.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setHDRNumberOfImages( int numberofimages );
	
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent state of this feature and an array of float values for the desired image. These float values represents the HDR parameters.
	\brief For a detailed description of these parameters, please look into the technical datasheet of the used camera.
	\param 'imageindex' (IN) The desired image index.
	\param 'pstate' (OUT) Current feature state.
	\param 'tParameters' (OUT) An array with current used HDR parameters.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getHDRParameters( int imageindex, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tParameters );

	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent state of this feature and an array of float values for the desired image. These float values represents the HDR parameters.
	\brief For a detailed description of these parameters, please look into the technical datasheet of the used camera.
	\param 'imageindex' (IN) The desired image index.
	\param 'parameter' (IN) The desired parameter.
	\param 'pstate' (OUT) Current feature state.
	\param 'tParameters' (OUT) An array with current used HDR parameters.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getHDRParameter( int imageindex, BGAPI_HDRParameter parameter, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tValue );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to set a HDR parameters for the desired image.
	\brief For a detailed description of the parameters, please look into the technical datasheet of the used camera.
	\param 'imageindex' (IN) The desired image index.
	\param 'parameter' (IN) The index into the parameter array of function getHDRParameters.
	\param 'value' (IN) The new HDR value.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setHDRParameter( int imageindex, BGAPI_HDRParameter parameter, float value );
		
	/**
	\brief Function category: BO Extended API.
	\brief This function returns the curent state of the reboot feature.
	\param 'pstate' (OUT) Current feature state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getReboot( BGAPI_FeatureState * pstate );

	/**
	\brief Function category: BO Extended API.
	\brief Use this function to reboot the camera, e.g. when changing the IP address of a gige camera.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT doReboot();

	/**
	\brief Function category: BO Image API.
	\brief Use this function to register a callback. The callback is a function pointer of the type BGAPI_NOTIFY_CALLBACK. This function is directly called by BGAPI, if the camera object is ready for image aquisition (see function Camera.setImage) and a new image was captured. In addition the user can pass a pointer, called a call back owner to determine the current context in the call back function.
	\param 'callBackOwner' (IN) Pointer to the user call back owner.
	\param 'callBack' (IN) Function pointer to a call back function.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT registerNotifyCallback( void * callBackOwner, BGAPI_NOTIFY_CALLBACK callBack );

	/**
	\brief Function category: BO Image API.
	\brief Use this function to register a callback to receive asynchronous messages. The callback is a function pointer of the type BGAPI_MESSAGE_CALLBACK. The user can pass a pointer, called a call back owner to determine the current context in the call back function.
	\param 'callBackOwner' (IN) Pointer to the user call back owner.
	\param 'callBack' (IN) Function pointer to a call back function.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT registerMessageCallback( void * callBackOwner, BGAPI_MESSAGE_CALLBACK callBack );

	/**
	\brief Function category: BO Image API.
	\brief Use this function to link an image object with the camera and it makes the camera ready for capturing. If an image was captured the call back function is called and the camera object lost the connection to the image. To capture a new one call this function again and pass the same image object or a new image object, created by function BGAPI::createImage to set a new camera - image link.
	\param 'imageObject' (IN) Desired BGAPI image.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setImage( Image * imageObject );

	/**
	\brief Function category: BO Image API.
	\brief Use this function to detach all images objects which are linked to the camera. This function works in queued buffer mode only if the camera is stopped (see setDataAccess).
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT detachImages();

	/**
	\brief Function category: Trace support.
	\brief This function creates a trace object to log some internal processes and states. 
	\param 'obj' (OUT) Object of trace class.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT createTraceObject( TraceObject ** obj );
};

/**
\brief Representatin of a BGAPI image object. 
*/
class BGAPI_DECL Image
{
private:
	/**
	\brief Function category: Constructor.
	\brief The constructor is a private function and can not directly called. To create an instance of this object use function BGAPI::createImage.
	*/
	Image();
    Image( const Image &Obj );
public:
	/**
	\brief Function category: BO Basic API.
	\brief This function return the count of supported BGAPI image features.
	\param 'count' (OUT) Get the count of Image features.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getFeatureCount( int * count );

	/**
	\brief Function category: BO Basic API.
	\brief This function executes a basic API feature.
	\param 'toset' (IN) The basic feature to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT setFeature( Feature * toset );

	/**
	\brief Function category: BO Basic API.
	\brief This function refreshs the basic API feature state and returns the actual settings.
	\param 'Id' (IN) Identifier of basic API feature to refresh.
	\param 'cur' (OUT) The pointer to desired basic API feature.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getFeature( BGAPI_FeatureId Id, Feature ** cur );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get a pointer to the begin of image data.
	\param 'imagebuffer' (OUT) Pointer to begin of image data.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT get( unsigned char ** imagebuffer );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get the user defined buffer, if it was set with function Image.setBuffer.
	\param 'externalbuffer' (OUT) Pointer of user buffer.
	\retval 'BGAPI_RESULT_OK'	No error.
	\retval 'BGAPI_RESULT_FAIL_MEMORY' No user allocated buffer was set with function Image.setBuffer.
	*/	
	BGAPI_RESULT getBuffer( unsigned char ** externalbuffer );

	/**
	\brief Function category: Set image information.
	\brief Use this function to set a user defined buffer.
	\param 'externalbuffer' (IN) Desired user buffer.
	\param 'length' (IN) The length of user buffer. Use the buffer size, which is returned by function Camera.getBufferLength.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setBuffer( unsigned char * externalbuffer, int length );

	/**
	\brief Function category: Retrieve image information.
	\brief Tests whether the buffer is an user buffer. 
	\param 'bextern' (OUT) True if the buffer is a user buffer.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT isExternBuffer( bool * bextern );

	/**
	\brief Function category: Set image information.
	\brief Use this function to load a image from file.
	\param 'filename' (IN) Name of image file.
	\param 'length' (IN) The length of file name.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setFromFile( char * filename, int length );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get the current buffer length in bytes.
	\param 'maxlength' (OUT) Maximum buffer length of all existing images formats of the connected camera.
	\param 'minlength' (OUT) Minimum necessary buffer length for the image. This buffer length based on the current image format settings of the connected camera and is byte exactly calculated.
	\param 'currentlength' (OUT) Current used buffer length.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getBufferLength( int * maxlength, int * minlength, int * currentlength );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get the image length in bytes.
	\param 'length' (OUT) Maximum buffer length.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getImageLength( int * length );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get the width and height in pixel.
	\param 'width' (OUT) Width of image.
	\param 'height' (OUT) Height of image.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getSize( int * width, int * height );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get the bit per pixel.
	\param 'bpp' (OUT) Used bits per pixel.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getBpp( int * bpp );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get the software generated counter and the hardware generated counter of this image.
	\param 'swcounter' (OUT) Software counter.
	\param 'hwcounter' (OUT) Hardware counter.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getNumber( int * swcounter, int * hwcounter );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get the timestamp of this image.
	\param 'timestamphigh' (OUT) Time Stamp. High part.
	\param 'timestamplow' (OUT) Time Stamp. Low Part.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getTimeStamp( int * timestamphigh, int * timestamplow );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get the BGAPI image header.
	\param 'header' (OUT) BGAPI image header.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getHeader( BGAPI_ImageHeader * header );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get the source pixel format of the image. Refer to definitions witch starting with BGAPI_PIXTYPE_.
	\param 'format' (OUT) Source pixel format.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getPixelFormat( int * format );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get the current image state.
	\param 'tImageStatus' (OUT) Current image status.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getState( BGAPI_ImageStatus * tImageStatus );

	/**
	\brief Function category: Retrieve image information.
	\brief Use this function to get detailed image informations.
	\param 'tImageInformation' (OUT) Detailed image informations.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getStateInformation( BGAPI_ImageInformation * tImageInformation );

	/**
	\brief Function category: Fundamental image conversion.
	\brief Use this function to get the size of the transformed image in bytes. You can use this value to allocate user buffer. Refer to function Image.setBuffer.
	\param 'length' (OUT) Length of transformed image.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getTransformBufferLength( int * length );

    /**
	\brief Function category: Fundamental image conversion.
	\brief This function starts the transformation process, including demosaic processes. First use the functions Image.setDestinationPixelFormat to configure the transorm process and than call this function.
	\param 'buffer' (OUT) Pointer to user allocated image buffer.
	\param 'length' (IN) The length of user buffer. Use the buffer size, which was returned by function Image.getTransformBufferLength.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT doTransform( unsigned char * buffer, int length );

    /**
	\brief Function category: Fundamental image conversion.
	\brief This function resets all transformation settings (e.g. destination pixelformat).
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT resetTransformSettings();

    /**
	\brief Function category: Fundamental image conversion.
	\brief This function selects the destination pixel format of the transformation process. See pixel type defines.
	\brief This function can be used in two several ways.
	\brief The first way is to use this function after you got the image via callback. Call this function and than Image.doTransform to start transformation.
	\brief The second way is only available for normal data access modes (refer to function Camera.setDataAccess and Camera.getDataAccess). Use this function directly after BGAPI::createImage. When the image is signaled via callback the transformation is already done. This way is faster than the first. But please note, by using a destination pixelformat which can not be transformed the original image data stays the same and the image state (function Image.getState) is set to BGAPI_IMAGESTATUS_NOTTRANSFORMED.
	\param 'pixelformat' (IN) The destination pixel format. Use the pixel type defines (BGAPI_PIXTYPE_).
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setDestinationPixelFormat( int pixelformat );

    /**
	\brief Function category: Fundamental image conversion.
	\brief This function returns the current state and a list with supported pixel formats, which depends of the specified 'sourcepixelformat'. The list returns only simple integer values. This integers have to interpret as pixel type defines (see BGAPI_PIXTYPE_).
	\param 'sourcepixelformat' (IN) The desired source pixel format.
	\param 'pstate' (OUT) Current state.
	\param 'tdest' (OUT) Supported destination pixel formats for transformation.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getDestinationPixelFormat( int sourcepixelformat, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tdest );

    /**
	\brief Function category: Fundamental image conversion.
	\brief Demosaic option, support of different demosaic operations.
	\param 'demosaic' (IN) Demosaic type to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setDemosaic( BGAPI_Demosaic demosaic );

    /**
	\brief Function category: Fundamental image conversion.
	\brief 	\brief This function returns the current state and a list with supported demosaic types. The list returns only simple integer values. This integers have to interpret as members of enumeration BGAPI_Demosaic.
	\param 'state' (OUT) Current state.
	\param 'list' (OUT) Current demosaic operation.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getDemosaic( BGAPI_FeatureState * state, BGAPIX_TypeListINT * list );
	
    /**
    \brief Function category: Retrieve image information.
    \brief 	This function returns a pointer to a chunk data block by a specified chunk id. A chunk data block can be the image block or the image header.
	\param 'chunkid' (IN) The desired chunk id.
	\param 'ppchunkofdata' (OUT) Pointer to chunk data block.
	\param 'length' (OUT) Length of chunk data block.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getChunkOfData( BGAPI_ChunkId chunkid, unsigned char ** ppchunkofdata, int * length );

	/**
	\brief Function category: Trace support.
	\brief This function creates a trace object to log some internal processes and states.
	\param 'obj' (OUT) Object of trace class.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/
	BGAPI_RESULT createTraceObject( TraceObject ** obj );
};

/**
\brief Representatin of a BGAPI feature object. BGAPI features are used with Basic API.
*/
class BGAPI_DECL Feature
{
private:
	/**
	\brief Function category: Constructor.
	\brief The constructor is a private function and can not directly called. To get a pointer of this object use function getFeature of class System, Camera and/or Image.
	*/
	Feature();
    Feature( const Feature & Obj );
public:
    /**
	\brief Use this function to get the feature id. Refer to enumeration BGAPI_FeatureId to get detailed informations about the supported feature id's.
	\param 'Id' (OUT) Current feature id.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getId( BGAPI_FeatureId * Id );

    /**
	\brief Use this function to get a brief feature description.
	\param 'Description' (OUT) Brief feature description.
	\param 'len' (IN) The length of 'Description' string.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getDescription( char * Description, int len );

    /**
	\brief Use this function to get the length of feature description in bytes.
	\param 'len' (OUT) The length of feature description in byte. 
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getDescriptionLength( int * len );

    /**
	\brief Use this function to get the current feature state.
	\param 'pstate' (OUT) Current state.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getState( BGAPI_FeatureState * pstate );

	/**
	\brief Use this function to get a next feature.
	\param 'next' (OUT) Pointer to the next linked feature.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getNext( Feature ** next );
    
    /**
	\brief Use this function to get a subordinate feature.
	\param 'sub' (OUT) Pointer to the subordinate feature.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getSub( Feature ** sub );

    /**
	\brief Use this function to get the current values of a feature.
	\param 'cur' (OUT) Pointer to the current values.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
    BGAPI_RESULT getValue( Value ** cur );

    /**
	\brief Use this function to get additional informations about a specified value of this feature. Refer to enumeration BGAPI_ParameterInfo for more informations.
	\param 'value' (IN) Desired BGAPI value. This BGAPI value must be a value of current feature.
	\param 'parainfo' (OUT) Additional informations of the BGAPI Value.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getParameterInfo( Value * value, BGAPI_ParameterInfo * parainfo );
};

/**
\brief Representatin of a BGAPI value object. BGAPI values are used with Basic API. 
*/
class BGAPI_DECL Value
{
private:
	/**
	\brief Function category: Constructor.
	\brief The constructor is a private function and can not directly called. To get a pointer of this object use function Feature.getValue.
	*/
    Value();
    Value( const Value & Obj );
public:
    /**
	\brief Use this function to get informations about the value type, value id and so on. Refer to enumeration BGAPI_ValueInfo to get detailed informations.
	\param 'valinfo' (OUT) Value informations.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getInfo( BGAPI_ValueInfo * valinfo );

    /**
	\brief Use this function to set a single boolean value.
	\param 'bval' (IN) Boolean value to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
    BGAPI_RESULT setBool( bool bval );

    /**
	\brief Use this function to get a single boolean value.
	\param 'bval' (OUT) Boolean value to get.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getBool( bool * bval );

    /**
	\brief Use this function to set an array with a maximum size of 32 bool values.
	\param 'bval' (IN) Bool array to set.
	\param 'length' (IN) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
    BGAPI_RESULT setArrayBool( bool bval[32], int length );

    /**
	\brief Use this function to get an array with a maximum size of 32 bool values.
	\param 'bval' (OUT) Bool array to get.
	\param 'length' (OUT) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getArrayBool( bool bval[32], int * length );

    /**
	\brief Use this function to set an large array with a maximum size of 4096 bool values.
	\param 'bval' (IN) Large bool array to set.
	\param 'length' (IN) Length of array. Maximum is 4096.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setLargeArrayBool( bool bval[4096], int length );

    /**
	\brief Use this function to get an large array with a maximum size of 4096 bool values.
	\param 'bval' (OUT) Large bool array to get.
	\param 'length' (OUT) Length of array. Maximum is 4096.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getLargeArrayBool( bool bval[4096], int * length );

    /**
	\brief Use this function to set a single integer value.
	\param 'ivalue' (IN) Integer value to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
    BGAPI_RESULT setInt( int ivalue );

    /**
	\brief Use this function to get a single integer value.
	\param 'ivalue' (OUT) Integer value to get.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getInt( int * ivalue );

    /**
	\brief Use this function to set an array with a maximum size of 32 integer values.
	\param 'ival' (IN) Integer array to set.
	\param 'length' (IN) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
    BGAPI_RESULT setArrayInt( int ival[32], int length );

    /**
	\brief Use this function to get an array with a maximum size of 32 integer values.
	\param 'ival' (OUT) Integer array to get.
	\param 'length' (OUT) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getArrayInt( int ival[32], int * length );

    /**
	\brief Use this function to set an large array with a maximum size of 4096 integer values.
	\param 'ival' (IN) Large integer array to set.
	\param 'length' (IN) Length of array. Maximum is 4096.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setLargeArrayInt( int ival[4096], int length );

    /**
	\brief Use this function to get an large array with a maximum size of 4096 integer values.
	\param 'ival' (OUT) Large integer array to get.
	\param 'length' (OUT) Length of array. Maximum is 4096.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getLargeArrayInt( int ival[4096], int * length );

    /**
	\brief Use this function to set a single float value.
	\param 'fval' (IN) Float value to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setFloat( float fval );

    /**
	\brief Use this function to get a single float value.
	\param 'fval' (OUT) Float value to get.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getFloat( float * fval );

    /**
	\brief Use this function to set an array with a maximum size of 32 float values.
	\param 'fval' (IN) Float array to set.
	\param 'length' (IN) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
    BGAPI_RESULT setArrayFloat( float fval[32], int length );

    /**
	\brief Use this function to get an array with a maximum size of 32 float values.
	\param 'fval' (OUT) Float array to get.
	\param 'length' (OUT) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getArrayFloat( float fval[32], int * length );

    /**
	\brief Use this function to set an large array with a maximum size of 4096 float values.
	\param 'fval' (IN) Large float array to set.
	\param 'length' (IN) Length of array. Maximum is 4096.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setLargeArrayFloat( float fval[4096], int length );

    /**
	\brief Use this function to get an large array with a maximum size of 4096 float values.
	\param 'fval' (OUT) Large float array to get.
	\param 'length' (OUT) Length of array. Maximum is 4096.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getLargeArrayFloat( float fval[4096], int * length );

    /**
	\brief Use this function to set a string with a specified length.
	\param 'sval' (IN) String to set.
	\param 'length' (IN) String length.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setString( char * sval, int length );

    /**
	\brief Use this function to get a string.
	\param 'sval' (OUT) String to get.
	\param 'length' (IN) String length.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getString( char * sval, int length );

    /**
	\brief Use this function to get the current string length and pass this value in function getString.
	\param 'length' (OUT) String length.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getStringLength( int * length );

    /**
	\brief Use this function to set a single byte value.
	\param 'chval' (IN) Byte to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setByte( unsigned char chval );

	/**
	\brief Use this function to get a single byte value.
	\param 'chval' (OUT) Byte to get.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getByte( unsigned char * chval );

    /**
	\brief Use this function to set an array with a maximum size of 32 bytes.
	\param 'ucval' (IN) Byte array to set.
	\param 'length' (IN) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setArrayByte( unsigned char ucval[32], int length );

    /**
	\brief Use this function to get an array with a maximum size of 32 bytes.
	\param 'ucval' (OUT) Byte array to get.
	\param 'length' (OUT) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getArrayByte( unsigned char ucval[32], int * length );

    /**
	\brief Use this function to set an large array with a maximum size of 4096 bytes.
	\param 'ucval' (IN) Large byte array to set.
	\param 'length' (IN) Length of array. Maximum is 4096.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setLargeArrayByte( unsigned char ucval[4096], int length );

    /**
	\brief Use this function to get an large array with a maximum size of 4096 bytes.
	\param 'ucval' (OUT) Large byte array to get.
	\param 'length' (OUT) Length of array. Maximum is 4096.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getLargeArrayByte( unsigned char ucval[4096], int * length );

	/**
	\brief Use this function to set a single unspecified pointer.
	\param 'ptr' (IN) Pointer to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
    BGAPI_RESULT setPtr( void * ptr );

    /**
	\brief Use this function to get a single unspecified pointer.
	\param 'ptr' (OUT) Pointer to get.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getPtr( void ** ptr );

    /**
	\brief Use this function to set a single handle.
	\param 'handle' (IN) Handle to set.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
    BGAPI_RESULT setHandle( void * handle );

    /**
	\brief Use this function to get a single handle.
	\param 'handle' (IN) Handle to get.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getHandle( void ** handle );

    /**
	\brief Use this function to set an array with a maximum size of 32 features id's.
	\param 'fid' (IN) Array with feature id's.
	\param 'length' (IN) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setArrayFeature( BGAPI_FeatureId fid[32], int length );

    /**
	\brief Use this function to get an array with a maximum size of 32 features id's.
	\param 'fid' (OUT) Array with feature id's.
	\param 'length' (OUT) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getArrayFeature( BGAPI_FeatureId fid[32], int * length );

    /**
	\brief Use this function to set an array with a maximum size of 32 values of class Value.
	\param 'vval' (IN) Array of Values.
	\param 'length' (IN) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setArrayValue( Value * vval[32], int length );

    /**
	\brief Use this function to get an array with a maximum size of 32 values of class Value.
	\param 'vval' (OUT) Array of Values.
	\param 'length' (OUT) Length of array. Maximum is 32.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getArrayValue( Value * vval[32], int * length );

    /**
	\brief Use this function to link values.
	\param 'next' (IN) Next value.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setNext( Value * next );

    /**
	\brief Use this function to get the linked value.
	\param 'next' (OUT) Next value.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getNext( Value ** next );

    /**
	\brief Use this function to get a list of values, which is associated to the passed value.
	\param 'parameterlist' (OUT) Parameter list.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getParameterList( Value ** parameterlist );

    /**
	\brief Use this function to get the first Value element of the parameterlist. This function is only usable for value typ BGAPI_VALUETYPE_LIST.
	\param 'pFirstElem' (OUT) The first parameterlist element of a value.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT getListValue( Value ** pFirstElem );
	
};

/**
\brief BGAPI Trace object. 
*/
class BGAPI_DECL TraceObject
{
private:
	/**
	\brief Function category: Constructor.
	\brief The constructor is a private function and can not directly called. To get a pointer of this object use function createTraceObject of class System, Camera and/or Image.
	*/
    TraceObject();
    TraceObject( const TraceObject & Obj );
public:
	/**
	\brief This function control the trace output for all BGAPI objects. The standard value is GLOBAL.
	\param 'ts' (IN) Set the trace options for all bgapi object (global), for all bgapi object of the same type or only for an instance. See enumeration BGAPI_TraceSupport.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setTraceSupport( BGAPI_TraceSupport ts );

	/**
	\brief This function enables the logging mechanism.
	\param 'bset' (IN) Enable/disable trace.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT enableLogging( bool bset );

	/**
	\brief This function sets the output options of the trace object.
	\param 'traceoutputoptions' (IN) Set output options. This values affect the form of the output string. See BGAPI_TRACEOUTPUTOPTION_XXX definitions. These definitions can be combiend.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setOutputOptions( int traceoutputoptions );

	/**
	\brief This function sets the trace target.
	\param 'target' (IN) Set the output target (see BGAPI_TraceTarget).
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setTarget( BGAPI_TraceTarget target );

	/**
	\brief This function set the trace filename, if the trace target BGAPI_TRACETARGET_FILE is selected.
	\param 'tracefilename' (IN) Set the output filename.
	\param 'size' (IN) String length in bytes.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setTraceFileName( char * tracefilename, int size );
	
	/**
	\brief This function set the trace mask.
	\param 'newmask' (IN) Set a mask to filter the output. See BGAPI_TRACEMASK_XXX definitions. These definitions can be combiend.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setMask( int newmask );

	/**
	\brief This function set the trace level.
	\param 'level' (IN) Set a trace level. See BGAPI_TRACELEVEL_XXX definitions.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT setLevel( int level );

	/**
	\brief This function register a callback function to receive trace outputs.
	\param 'callBackOwner' (IN) Pointer to the user call back owner.
	\param 'traceCallBack' (IN) Function pointer to a call back function.
	\retval 'BGAPI_RESULT_OK'	No error.
	*/	
	BGAPI_RESULT registerTraceCallback( void * callBackOwner, BGAPI_TRACE_CALLBACK traceCallBack );
};
}

#endif //_BGAPI_HPP__
