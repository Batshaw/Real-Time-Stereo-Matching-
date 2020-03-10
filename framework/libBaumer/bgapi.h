/*************************************************************
         Baumer Optronic C Interface for bgapi
**************************************************************/

#if !defined(_GNULINUX)
#ifdef BGAPI_EXPORTS
#define BGAPI_DECL __declspec(dllexport)
#else
#define BGAPI_DECL __declspec(dllimport)
#endif
#else
#define BGAPI_DECL
#endif
// Calling Conventions
#if defined(__BORLANDC__) || (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)
    #define BGAPICALL __stdcall
#else
    #define BGAPICALL
#endif

#ifndef _BGAPI_H__
#define _BGAPI_H__

typedef char BGAPI_bool;
#define BGAPI_false	0x00
#define BGAPI_true	0x01

#include "bgapiresult.h"
#include "bgapidef.h"

#ifdef __cplusplus
extern "C"
{
#endif


typedef void (BGAPI_CALLBACK *BGAPI_NOTIFY_CALLBACK_C)( void * callBackOwner, BGAPI_Image * image );

typedef void (BGAPI_CALLBACK *BGAPI_MESSAGE_CALLBACK_C)( void * callBackOwner, BGAPI_Message * message );

typedef void (BGAPI_CALLBACK *BGAPI_TRACE_CALLBACK_C)( void * callBackOwner, char ** tracemsg );

typedef void (BGAPI_CALLBACK *BGAPI_PNP_CALLBACK_C)( void * callBackOwner, BGAPI_PnPEvent * pnpevent );

typedef void (BGAPI_CALLBACK *BGAPI_BUSRESET_CALLBACK_C)( void * callBackOwner );



/* entry points to bgapi */
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_countSystems( int * count );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_createSystem( int index, BGAPI_System ** system );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_getBGAPIVersionLength( int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_getBGAPIVersion( char * bgapiversion, int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_setModuleSearchPath( char * path, int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_createImage( BGAPI_Image ** img );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_releaseImage( BGAPI_Image * img );

BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_open( BGAPI_System * system );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_release( BGAPI_System * system );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_countCameras( BGAPI_System * system, int* count );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_createCamera( BGAPI_System * system, int index, BGAPI_Camera ** cam );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_releaseCamera( BGAPI_System * system, BGAPI_Camera * cam );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_createImage( BGAPI_System * system, BGAPI_Image ** img, BGAPI_bool useexternbuffer );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_releaseImage( BGAPI_System * system, BGAPI_Image * img );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_getFeatureCount( BGAPI_System * system, int* count );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_setFeature( BGAPI_System * system, BGAPI_Feature * toset );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_getFeature( BGAPI_System * system, BGAPI_FeatureId Id, BGAPI_Feature ** cur );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_getDeviceInformation( BGAPI_System * system, BGAPI_FeatureState * pstate, BGAPIX_SystemInfo * deviceinfo );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_setGVSDriverModel( BGAPI_System * system, BGAPI_GVSDriverModel drivermodel );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_getGVSDriverModel( BGAPI_System * system, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_doActionCommand( BGAPI_System * system, unsigned int devicekey, unsigned int groupkey, unsigned int groupmask, BGAPI_bool bWaitForAnswer );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_getActionCommand( BGAPI_System * system, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_doGVSForceIP( BGAPI_System * system, unsigned char mac[6], unsigned char ip[32], unsigned char subnetmask[32], unsigned char gateway[32], int size );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_doCommandResponderListAdd( BGAPI_System * system, unsigned char device[32] );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_doCommandResponderListReset( BGAPI_System * system );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_getNumberOfFailedCommandResponder( BGAPI_System * system, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tNumberOf );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_getFailedCommandResponder( BGAPI_System * system, BGAPI_FeatureState * pstate, int index, BGAPIX_TypeArrayBYTE * tDevice );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_getNumberOfAvailableInterfaces( BGAPI_System * system, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tNumberOf );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_getAvailableInterface( BGAPI_System * system, int index, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayBYTE * tInterface, BGAPIX_TypeArrayBYTE * tMask );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_getNumberOfIgnoredInterfaces( BGAPI_System * system, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tNumberOf );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_getIgnoredInterface( BGAPI_System * system, int index, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayBYTE * tInterface );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_setInterfaceToIgnore( BGAPI_System * system, unsigned char interfacetoignore[32] );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_doInterfaceToIgnoreListReset( BGAPI_System * system );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_createTraceObject( BGAPI_System * system, BGAPI_TraceObject ** obj );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_registerPnPCallback( BGAPI_System * system, void * callBackOwner, BGAPI_PNP_CALLBACK_C PnPcallBack );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_System_registerBusResetCallback( BGAPI_System * system, void * callBackOwner, BGAPI_BUSRESET_CALLBACK_C BusResetCallBack );

BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_open( BGAPI_Camera * camera );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getFeatureCount( BGAPI_Camera * camera, int* count );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setFeature( BGAPI_Camera * camera, BGAPI_Feature * toset );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getFeature( BGAPI_Camera * camera, BGAPI_FeatureId Id, BGAPI_Feature ** cur );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setImageFormat( BGAPI_Camera * camera, int formatindex );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getImageFormat( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getImageFormatDescription( BGAPI_Camera * camera, int formatindex, BGAPIX_CameraImageFormat * cformat );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getFormatFeature( BGAPI_Camera * camera, int formatindex, BGAPIX_TypeListFEATURE * flist );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setPixelFormat( BGAPI_Camera * camera, int pixelformat );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getPixelFormat( BGAPI_Camera * camera, int formatindex, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getPixelFormatDescription( BGAPI_Camera * camera, int formatindex, int pixelformat, BGAPIX_CameraPixelFormat * pformat );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setTrigger( BGAPI_Camera * camera, BGAPI_bool enable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTrigger( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setTriggerSource( BGAPI_Camera * camera, BGAPI_TriggerSource trigger_source );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTriggerSource( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tSource );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setTriggerActivation( BGAPI_Camera * camera, BGAPI_Activation activation );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTriggerActivation( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tActivation );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setTriggerDelay( BGAPI_Camera * camera, int delay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTriggerDelay( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDelay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_doTrigger( BGAPI_Camera * camera );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setFlash( BGAPI_Camera * camera, BGAPI_bool enable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getFlash( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setFlashActivation( BGAPI_Camera * camera, BGAPI_Activation activation );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getFlashActivation( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tActivation );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setFlashDelay( BGAPI_Camera * camera, int flashdelay, int triggerdelay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getFlashDelay( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tFlashDelay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setCooler( BGAPI_Camera * camera, BGAPI_bool enable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getCooler( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setAEC( BGAPI_Camera * camera, BGAPI_bool enable, int brightpercent, int left, int top, int right, int bottom );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getAEC( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * brightpercent, BGAPIX_TypeROI * roi );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setExposure( BGAPI_Camera * camera, int exposure );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getExposure( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * sint );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setShutter( BGAPI_Camera * camera, BGAPI_Shutter shutter );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getShutter( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGain( BGAPI_Camera * camera, float gain );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGain( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * sfloat );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setDigitalGain( BGAPI_Camera * camera, int dgain );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getDigitalGain( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDigitalGain );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setOffset( BGAPI_Camera * camera, int offset );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getOffset( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * sint );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setPartialScan( BGAPI_Camera * camera, BGAPI_bool enable, int left, int top, int right, int bottom );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getPartialScan( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeROI * roi );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_doWhiteBalance( BGAPI_Camera * camera, int left, int top, int right, int bottom );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setWhiteBalance( BGAPI_Camera * camera, BGAPI_bool enable, float red, float green, float blue );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getWhiteBalance( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * wbred, BGAPIX_TypeRangeFLOAT * wbgreen, BGAPIX_TypeRangeFLOAT * wbblue, BGAPIX_TypeROI * roi );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_doFactorySettings( BGAPI_Camera * camera );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getFactorySettings( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setStart( BGAPI_Camera * camera, BGAPI_bool enable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getStart( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setTestPattern( BGAPI_Camera * camera, BGAPI_bool enable, BGAPI_TestPattern mode );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTestPattern( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_doAsyncReset( BGAPI_Camera * camera, BGAPI_bool delay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getAsyncReset( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setReadout( BGAPI_Camera * camera, BGAPI_Readout readout );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getReadout( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setDataAccess( BGAPI_Camera * camera, BGAPI_DataAccess mode, int numberofbuffer, BGAPI_bool useexternbuffer );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getDataAccess( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * mode, BGAPIX_TypeINT * tnumberofbuffer, BGAPIX_TypeBOOL * tuseexternbuffer );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setDataAccessMode( BGAPI_Camera * camera, BGAPI_DataAccessMode tMode, int numberofbuffer );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getDataAccessMode( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list, BGAPIX_TypeINT * tnumberofbuffer );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setFrameCounter( BGAPI_Camera * camera, int swcounter, int hwcounter );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getFrameCounter( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * swrange, BGAPIX_TypeRangeINT * hwrange );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_resetTimeStamp( BGAPI_Camera * camera );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTimeStamp( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * timestamphigh, BGAPIX_TypeINT * timestamplow, BGAPIX_TypeINT * frequencyhigh, BGAPIX_TypeINT * frequencylow );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setCanalSelect( BGAPI_Camera * camera, int canal );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getCanalSelect( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT *list );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLookupTableGamma( BGAPI_Camera * camera, BGAPI_bool enable, int red[4096], int green[4096], int blue[4096], int size );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLookupTableGamma( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeLargeArrayINT * r, BGAPIX_TypeLargeArrayINT * g, BGAPIX_TypeLargeArrayINT * b );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setCommand( BGAPI_Camera * camera, char * pcommand, int size, BGAPIX_TypeSTRING * response );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getCommand( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setBrightnessCorrection( BGAPI_Camera * camera, BGAPI_bool bSet );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getBrightnessCorrection( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getDeviceInformation( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_CameraInfo * deviceinfo );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setUserId( BGAPI_Camera * camera, char * userid, int size );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getUserId( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeSTRING * tuserid );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGVSHeartBeatEnable( BGAPI_Camera * camera, BGAPI_bool benable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGVSHeartBeatEnable( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGVSHeartBeatTimeout( BGAPI_Camera * camera, int htimeout );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGVSHeartBeatTimeout( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * timeinms );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGVSPacketDelay( BGAPI_Camera * camera, int packetdelay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGVSPacketDelay( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tPacketDelay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGVSPersistentIP( BGAPI_Camera * camera, BGAPI_bool enable, unsigned char ip[32], unsigned char subnetmask[32], unsigned char gateway[32], int size );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGVSPersistentIP( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeBOOL * tenable, BGAPIX_TypeArrayBYTE * tip, BGAPIX_TypeArrayBYTE * tsubnetmask, BGAPIX_TypeArrayBYTE * tgateway );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGVSDHCP( BGAPI_Camera * camera, BGAPI_bool enable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGVSDHCP( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeBOOL * tenable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setPacketSize( BGAPI_Camera * camera, int packetsize );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getPacketSize( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tpacketsize );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setFramesPerSecondsContinuous( BGAPI_Camera * camera, float framerate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getFramesPerSecondsContinuous( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tframerate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setFramesPerSeconds( BGAPI_Camera * camera, float framerate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getFramesPerSeconds( BGAPI_Camera * camera, int formatindex, BGAPI_FeatureState * pstate, BGAPIX_TypeListFLOAT * tframerate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_resetStatistic( BGAPI_Camera * camera );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getStatistic( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_CameraStatistic * statistic );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setSequencer( BGAPI_Camera * camera, BGAPI_bool enable, BGAPI_bool start );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSequencer( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeBOOL * pstart );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setSequencerRunMode( BGAPI_Camera * camera, BGAPI_SequencerRunMode runmode );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSequencerRunMode( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * prunmodelist );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_doSequencerAddParameter( BGAPI_Camera * camera );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSequencerAddParameter( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * pnumofparamsets );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_doSequencerReset( BGAPI_Camera * camera );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSequencerReset( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setSequencerCounter( BGAPI_Camera * camera, int loopcounter, int repeatcounter, int triggercounter );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSequencerCounter( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * ploopcounter, BGAPIX_TypeRangeINT * prepeatcounter, BGAPIX_TypeRangeINT * ptriggercounter );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setSequencerExposure( BGAPI_Camera * camera, int exposure );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSequencerExposure( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tExposure );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setSequencerGain( BGAPI_Camera * camera, float gain );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSequencerGain( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tGain );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setSequencerPWMDutyCycle( BGAPI_Camera * camera, BGAPI_PWM pwm, float dutycycle );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSequencerPWMDutyCycle( BGAPI_Camera * camera, BGAPI_PWM pwm, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tDutyCycle );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getReadoutTime( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tReadoutTime );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLines( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAvailable, BGAPIX_TypeINT * tCurrent );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getAvailableLines( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAvailable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineState( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPI_Polarity * polarity, BGAPIX_TypeListINT * tMode, BGAPIX_TypeListINT * tOutputSource );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineState( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_LineMode mode, BGAPI_LineOutputSource outputsource );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineMode( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tMode );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineMode( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_LineMode mode );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineOutputSource( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tOutputSource );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineOutputSource( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_LineOutputSource outputsource );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineDebouncer( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT *tLowTime, BGAPIX_TypeRangeINT * tHighTime );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineDebouncerLow( BGAPI_Camera * camera, BGAPI_Line line, int lowtime );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineDebouncerHigh( BGAPI_Camera * camera, BGAPI_Line line, int hightime );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSingleOutput( BGAPI_Camera * camera, BGAPI_UserOutput useroutput, BGAPI_FeatureState * pstate, BGAPI_Polarity * polarity );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setSingleOutput( BGAPI_Camera * camera, BGAPI_UserOutput useroutput, BGAPI_Polarity polarity );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getParallelOutput( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAvailable, BGAPIX_TypeINT * tCurrent );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setParallelOutput( BGAPI_Camera * camera, int newstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getInputLineMask( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tInputMask );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getOutputLineMask( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tOutputMask );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getMessage( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAvailable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setMessage( BGAPI_Camera * camera, BGAPI_MessageId messageid, BGAPI_MessageState messagestate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getMessageState( BGAPI_Camera * camera, BGAPI_MessageId messageid, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tMessageState );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getDefectPixelCorrection( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setDefectPixelCorrection( BGAPI_Camera * camera, BGAPI_bool enable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getDefectPixels( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tCurrentIndex, BGAPIX_TypeLargeArrayINT * tX, BGAPIX_TypeINT *tXMin, BGAPIX_TypeINT *tXMax, BGAPIX_TypeLargeArrayINT * tY, BGAPIX_TypeINT *tYMin, BGAPIX_TypeINT *tYMax, BGAPIX_TypeLargeArrayBOOL * tEnabled );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setDefectPixel( BGAPI_Camera * camera, int index, int x, int y, BGAPI_bool enable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getUserSet( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tUserSet );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setUserSet( BGAPI_Camera * camera, BGAPI_UserSet userset );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_doUserSetStore( BGAPI_Camera * camera, BGAPI_UserSet userset );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getUserSetDefault( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tDefaultUserSet );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setUserSetAsDefault( BGAPI_Camera * camera, BGAPI_UserSet userset );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGamma( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tGamma );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGamma( BGAPI_Camera * camera, float gamma );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLookUpTable( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tLUT );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLookUpTableState( BGAPI_Camera * camera, BGAPI_LookUpTable lut, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLookUpTableState( BGAPI_Camera * camera, BGAPI_LookUpTable lut, BGAPI_bool enable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLookUpTableValues( BGAPI_Camera * camera, BGAPI_LookUpTable lut, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tCurrentIndex, BGAPIX_TypeLargeArrayINT * tLUTValues, BGAPIX_TypeINT *tLUTMin, BGAPIX_TypeINT *tLUTMax );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLookUpTableValue( BGAPI_Camera * camera, BGAPI_LookUpTable lut, int index, int lutvalue );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTemperatureSensors( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tTemperatureSensors );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTemperature( BGAPI_Camera * camera, BGAPI_TemperatureSensor temperaturesensor, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tCurrentTemperature );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTemperatureCritical( BGAPI_Camera * camera, BGAPI_TemperatureSensor temperaturesensor, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tCriticalTemperature );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setTemperatureCritical( BGAPI_Camera * camera, BGAPI_TemperatureSensor temperaturesensor, float criticaltemperature );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getPWM( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tPWM );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getPWMEnable( BGAPI_Camera * camera, BGAPI_PWM pwm, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setPWMEnable( BGAPI_Camera * camera, BGAPI_PWM pwm, BGAPI_bool enable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getPWMFrequency( BGAPI_Camera * camera, BGAPI_PWM pwm, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tFrequency );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setPWMFrequency( BGAPI_Camera * camera, BGAPI_PWM pwm, float frequency );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getPWMDutyCycle( BGAPI_Camera * camera, BGAPI_PWM pwm, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tDutyCycle );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setPWMDutyCycle( BGAPI_Camera * camera, BGAPI_PWM pwm, float dutycycle );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getPWMOverPowerThreshold( BGAPI_Camera * camera, BGAPI_PWM pwm, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tOverPowerThreshold );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setPWMOverPowerThreshold( BGAPI_Camera * camera, BGAPI_PWM pwm, float overpowerthreshold );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getDistanceAmplitudeThreshold( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * threshold );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setDistanceAmplitudeThreshold( BGAPI_Camera * camera, int threshold );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getToFDistanceAmplitudeThreshold( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * threshold );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setToFDistanceAmplitudeThreshold( BGAPI_Camera * camera, int threshold );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getToFModulationFrequency( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tFrequency );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setToFModulationFrequency( BGAPI_Camera * camera, BGAPI_ToFModulationFrequency frequency );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLinePotential( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * list );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLinePotential( BGAPI_Camera * camera, BGAPI_LinePotential linePotential );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getDataLength( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDataLength );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getImageResolution( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tWidth, BGAPIX_TypeRangeINT * tHeight );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setMulticast( BGAPI_Camera * camera, BGAPI_bool benable, unsigned char multicastip[32] );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getMulticast( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayBYTE * tMulticastIP );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setMulticastMessage( BGAPI_Camera * camera, BGAPI_bool benable, unsigned char multicastmessageip[32] );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getMulticastMessage( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayBYTE * tMulticastMessageIP );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getConfigMode( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeBOOL * tEnable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setConfigMode( BGAPI_Camera * camera, BGAPI_bool bEnable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineActivation( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tActivation );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineActivation( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_Activation tActivation );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineDelay( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDelay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineDelay( BGAPI_Camera * camera, BGAPI_Line line, unsigned int delay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineDelayTimeBase( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tTimeBaseLines );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineDelayTimeBase( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_Line timebaseline );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineDelayClockRateDivider( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tClockRateDivider );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineDelayClockRateDivider( BGAPI_Camera * camera, BGAPI_Line line, int clockratedivider );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getActionCommands( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tActionCommands );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getActionCommand( BGAPI_Camera * camera, unsigned int actioncommand, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tGroupKey, BGAPIX_TypeINT * tGroupMask );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setActionCommand( BGAPI_Camera * camera, unsigned int actioncommand, unsigned int groupkey, unsigned int groupmask );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setActionDeviceKey( BGAPI_Camera * camera, unsigned int devicekey );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSenderActionCommands( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tActionCommands );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSenderActionCommand( BGAPI_Camera * camera, unsigned int actioncommand, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tGroupKey, BGAPIX_TypeINT * tGroupMask );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setSenderActionCommand( BGAPI_Camera * camera, unsigned int actioncommand, unsigned int devicekey, unsigned int groupkey, unsigned int groupmask );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSenderNumberOfResponder( BGAPI_Camera * camera, unsigned int actioncommand, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tNumberOf );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSenderResponder( BGAPI_Camera * camera, unsigned int actioncommand, int index, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayBYTE * tResponder );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_doSenderResponderListAdd( BGAPI_Camera * camera, unsigned int actioncommand, unsigned char responder[32] );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_doSenderResponderListReset( BGAPI_Camera * camera, unsigned int actioncommand );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getSenderAbortConditions( BGAPI_Camera * camera, unsigned int actioncommand, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tTimeout, BGAPIX_TypeINT * tRetry );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setSenderAbortConditions( BGAPI_Camera * camera, unsigned int actioncommand, int timeout, int retry );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLinePulseDelay( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tPulseDelay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLinePulseDelay( BGAPI_Camera * camera, BGAPI_Line line, unsigned int pulsedelay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLinePulseDelayTimeBase( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tTimeBaseLines );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLinePulseDelayTimeBase( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_Line timebaseline );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLinePulseDelayClockRateDivider( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tClockRateDivider );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLinePulseDelayClockRateDivider( BGAPI_Camera * camera, BGAPI_Line line, int clockratedivider );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLinePulseWidth( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tPulseWidth );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLinePulseWidth( BGAPI_Camera * camera, BGAPI_Line line, unsigned int pulsewidth );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLinePulseWidthTimeBase( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tTimeBaseLines );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLinePulseWidthTimeBase( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_Line timebaseline );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLinePulseWidthClockRateDivider( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tClockRateDivider );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLinePulseWidthClockRateDivider( BGAPI_Camera * camera, BGAPI_Line line, int clockratedivider );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineGate( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tGateMode );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineGate( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_LineGate gatemode );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineReTriggerTime( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tReTriggerTime );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineReTriggerTime( BGAPI_Camera * camera, BGAPI_Line line, unsigned int retriggertime );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineReTriggerTimeBase( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tTimeBaseLines );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineReTriggerTimeBase( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_Line timebaseline );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineReTriggerClockRateDivider( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeINT * tClockRateDivider );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineReTriggerClockRateDivider( BGAPI_Camera * camera, BGAPI_Line line, int clockratedivider );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getLineInverter( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_FeatureState * pstate, BGAPIX_TypeBOOL * tInverted );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setLineInverter( BGAPI_Camera * camera, BGAPI_Line line, BGAPI_bool binvert );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getFlipType( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tFlipType );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setFlipType( BGAPI_Camera * camera, BGAPI_FlipType fliptype );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setExposureAuto( BGAPI_Camera * camera, BGAPI_ProcessingUnit processingunit );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getExposureAuto( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tProcessingUnit );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setExposureAutoMode( BGAPI_Camera * camera, BGAPI_AutomaticMode automode );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getExposureAutoMode( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAutoMode );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setExposureAutoBrightness( BGAPI_Camera * camera, float brightinpercent );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getExposureAutoBrightness( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tBrightInPercent );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setExposureAutoROI( BGAPI_Camera * camera, int left, int top, int right, int bottom );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getExposureAutoROI( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeROI *tROI );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setExposureAutoMinimum( BGAPI_Camera * camera, int minexposure );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getExposureAutoMinimum( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tMinExposure );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setExposureAutoMaximum( BGAPI_Camera * camera, int maxexposure );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getExposureAutoMaximum( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tMaxExposure );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getOffsetL( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tOffsetL );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setOffsetL( BGAPI_Camera * camera, int offsetL );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getOffseta( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tOffseta );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setOffseta( BGAPI_Camera * camera, int offseta );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getOffsetb( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tOffsetb );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setOffsetb( BGAPI_Camera * camera, int offsetb );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGainL( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tGainL );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGainL( BGAPI_Camera * camera, float gainL );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGaina( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tGaina );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGaina( BGAPI_Camera * camera, float gaina );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGainb( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tGainb );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGainb( BGAPI_Camera * camera, float gainb );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setTransmissionDelay( BGAPI_Camera * camera, int delay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTransmissionDelay( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDelay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setScalingFactorX( BGAPI_Camera * camera, float xfactor );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getScalingFactorX( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tFactorX );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setScalingFactorY( BGAPI_Camera * camera, float yfactor );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getScalingFactorY( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tFactorY );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTimers( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tTimer );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTimerDuration( BGAPI_Camera * camera, BGAPI_Timer timer, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDuration );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setTimerDuration( BGAPI_Camera * camera, BGAPI_Timer timer, int duration );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTimerDelay( BGAPI_Camera * camera, BGAPI_Timer timer, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tDelay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setTimerDelay( BGAPI_Camera * camera, BGAPI_Timer timer, int delay );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTimerTriggerSource( BGAPI_Camera * camera, BGAPI_Timer timer, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tSource );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setTimerTriggerSource( BGAPI_Camera * camera, BGAPI_Timer timer, BGAPI_TriggerSource source );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getTimerTriggerActivation( BGAPI_Camera * camera, BGAPI_Timer timer, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tActivation );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setTimerTriggerActivation( BGAPI_Camera * camera, BGAPI_Timer timer, BGAPI_Activation activation );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGainAutoMode( BGAPI_Camera * camera, BGAPI_AutomaticMode automode );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGainAutoMode( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tAutoMode );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setGainAutoMaximum( BGAPI_Camera * camera, float maxgain );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getGainAutoMaximum( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tMaxGain );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getCounters( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tCounter );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getCounter( BGAPI_Camera * camera, BGAPI_Counter counter, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tValue );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setCounter( BGAPI_Camera * camera, BGAPI_Counter counter, int value );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getCounterTriggerSource( BGAPI_Camera * camera, BGAPI_Counter counter, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tSource );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setCounterTriggerSource( BGAPI_Camera * camera, BGAPI_Counter counter, BGAPI_TriggerSource source );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getCounterTriggerActivation( BGAPI_Camera * camera, BGAPI_Counter counter, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tActivation );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setCounterTriggerActivation( BGAPI_Camera * camera, BGAPI_Counter counter, BGAPI_Activation activation );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getHDRProcessingUnit( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tProcessingUnit );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setHDRProcessingUnit( BGAPI_Camera * camera, BGAPI_ProcessingUnit processingunit );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getHDREnable( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setHDREnable( BGAPI_Camera * camera, BGAPI_bool benable );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getHDRNumberOfImages( BGAPI_Camera * camera, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeINT * tNumberOfImages );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setHDRNumberOfImages( BGAPI_Camera * camera, int numberofimages );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getHDRParameters( BGAPI_Camera * camera, int imageindex, BGAPI_FeatureState * pstate, BGAPIX_TypeArrayINT * tParameters );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getHDRParameter( BGAPI_Camera * camera, int imageindex, BGAPI_HDRParameter parameter, BGAPI_FeatureState * pstate, BGAPIX_TypeRangeFLOAT * tValue );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setHDRParameter( BGAPI_Camera * camera, int imageindex, BGAPI_HDRParameter parameter, float value );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_getReboot( BGAPI_Camera * camera, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_doReboot( BGAPI_Camera * camera );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_registerNotifyCallback( BGAPI_Camera * camera, void * callBackOwner, BGAPI_NOTIFY_CALLBACK_C callBack );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_registerMessageCallback( BGAPI_Camera * camera, void * callBackOwner, BGAPI_MESSAGE_CALLBACK_C callBack );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_setImage( BGAPI_Camera * camera, BGAPI_Image * imageObject );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_detachImages( BGAPI_Camera * camera );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Camera_createTraceObject( BGAPI_Camera * camera, BGAPI_TraceObject ** obj );

BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getFeatureCount( BGAPI_Image * image, int * count );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_setFeature( BGAPI_Image * image, BGAPI_Feature * toset );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getFeature( BGAPI_Image * image, BGAPI_FeatureId Id, BGAPI_Feature ** cur );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_get( BGAPI_Image * image, unsigned char ** imagebuffer );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getBuffer( BGAPI_Image * image, unsigned char ** externalbuffer );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_setBuffer( BGAPI_Image * image, unsigned char * externalbuffer, int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_isExternBuffer( BGAPI_Image * image, BGAPI_bool * bextern );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_setFromFile( BGAPI_Image * image, char * filename, int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getBufferLength( BGAPI_Image * image, int * maxlength, int * minlength, int * currentlength );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getImageLength( BGAPI_Image * image, int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getSize( BGAPI_Image * image, int * width, int * height );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getBpp( BGAPI_Image * image, int * bpp );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getNumber( BGAPI_Image * image, int * swcounter, int * hwcounter );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getTimeStamp( BGAPI_Image * image, int * timestamphigh, int * timestamplow );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getHeader( BGAPI_Image * image, BGAPI_ImageHeader * header );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getPixelFormat( BGAPI_Image * image, int * format );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getState( BGAPI_Image * image, BGAPI_ImageStatus * tImageStatus );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getStateInformation( BGAPI_Image * image, BGAPI_ImageInformation * tImageInformation );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getTransformBufferLength( BGAPI_Image * image, int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_doTransform( BGAPI_Image * image, unsigned char * buffer, int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_resetTransformSettings( BGAPI_Image * image );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_setDestinationPixelFormat( BGAPI_Image * image, int pixelformat );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getDestinationPixelFormat( BGAPI_Image * image, int sourcepixelformat, BGAPI_FeatureState * pstate, BGAPIX_TypeListINT * tdest );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_setDemosaic( BGAPI_Image * image, BGAPI_Demosaic demosaic );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getDemosaic( BGAPI_Image * image, BGAPI_FeatureState * state, BGAPIX_TypeListINT * list );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_getChunkOfData( BGAPI_Image * image, BGAPI_ChunkId chunkid, unsigned char ** ppchunkofdata, int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Image_createTraceObject( BGAPI_Image * image, BGAPI_TraceObject ** obj );

BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Feature_getId( BGAPI_Feature * feature, BGAPI_FeatureId * Id );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Feature_getDescription( BGAPI_Feature * feature, char * Description, int len );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Feature_getDescriptionLength( BGAPI_Feature * feature, int * len );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Feature_getState( BGAPI_Feature * feature, BGAPI_FeatureState * pstate );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Feature_getNext( BGAPI_Feature * feature, BGAPI_Feature ** next );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Feature_getSub( BGAPI_Feature * feature, BGAPI_Feature ** sub );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Feature_getValue( BGAPI_Feature * feature, BGAPI_Value ** cur );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Feature_getParameterInfo( BGAPI_Feature * feature, BGAPI_Value * value, BGAPI_ParameterInfo * parainfo );

BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getInfo( BGAPI_Value * value, BGAPI_ValueInfo * valinfo );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setBool( BGAPI_Value * value, BGAPI_bool bval );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getBool( BGAPI_Value * value, BGAPI_bool * bval );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setArrayBool( BGAPI_Value * value, BGAPI_bool bval[32], int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getArrayBool( BGAPI_Value * value, BGAPI_bool bval[32], int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setLargeArrayBool( BGAPI_Value * value, BGAPI_bool bval[4096], int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getLargeArrayBool( BGAPI_Value * value, BGAPI_bool bval[4096], int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setInt( BGAPI_Value * value, int ivalue );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getInt( BGAPI_Value * value, int * ivalue );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setArrayInt( BGAPI_Value * value, int ival[32], int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getArrayInt( BGAPI_Value * value, int ival[32], int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setLargeArrayInt( BGAPI_Value * value, int ival[4096], int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getLargeArrayInt( BGAPI_Value * value, int ival[4096], int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setFloat( BGAPI_Value * value, float fval );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getFloat( BGAPI_Value * value, float * fval );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setArrayFloat( BGAPI_Value * value, float fval[32], int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getArrayFloat( BGAPI_Value * value, float fval[32], int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setLargeArrayFloat( BGAPI_Value * value, float fval[4096], int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getLargeArrayFloat( BGAPI_Value * value, float fval[4096], int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setString( BGAPI_Value * value, char * sval, int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getString( BGAPI_Value * value, char * sval, int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getStringLength( BGAPI_Value * value, int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setByte( BGAPI_Value * value, unsigned char chval );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getByte( BGAPI_Value * value, unsigned char * chval );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setArrayByte( BGAPI_Value * value, unsigned char ucval[32], int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getArrayByte( BGAPI_Value * value, unsigned char ucval[32], int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setLargeArrayByte( BGAPI_Value * value, unsigned char ucval[4096], int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getLargeArrayByte( BGAPI_Value * value, unsigned char ucval[4096], int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setPtr( BGAPI_Value * value, void * ptr );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getPtr( BGAPI_Value * value, void ** ptr );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setHandle( BGAPI_Value * value, void * handle );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getHandle( BGAPI_Value * value, void ** handle );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setArrayFeature( BGAPI_Value * value, BGAPI_FeatureId fid[32], int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getArrayFeature( BGAPI_Value * value, BGAPI_FeatureId fid[32], int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setArrayValue( BGAPI_Value * value, BGAPI_Value * vval[32], int length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getArrayValue( BGAPI_Value * value, BGAPI_Value * vval[32], int * length );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_setNext( BGAPI_Value * value, BGAPI_Value * next );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getNext( BGAPI_Value * value, BGAPI_Value ** next );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getParameterList( BGAPI_Value * value, BGAPI_Value ** parameterlist );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_Value_getListValue( BGAPI_Value * value, BGAPI_Value ** pFirstElem );

BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_TraceObject_setTraceSupport( BGAPI_TraceObject * traceobject, BGAPI_TraceSupport ts );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_TraceObject_enableLogging( BGAPI_TraceObject * traceobject, BGAPI_bool bset );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_TraceObject_setOutputOptions( BGAPI_TraceObject * traceobject, int traceoutputoptions );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_TraceObject_setTarget( BGAPI_TraceObject * traceobject, BGAPI_TraceTarget target );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_TraceObject_setTraceFileName( BGAPI_TraceObject * traceobject, char * tracefilename, int size );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_TraceObject_setMask( BGAPI_TraceObject * traceobject, int newmask );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_TraceObject_setLevel( BGAPI_TraceObject * traceobject, int level );
BGAPI_DECL BGAPI_RESULT BGAPICALL BGAPI_TraceObject_registerTraceCallback( BGAPI_TraceObject * traceobject, void * callBackOwner, BGAPI_TRACE_CALLBACK_C traceCallBack );


#ifdef __cplusplus
} //extern "C"
#endif

#endif //_BGAPI_H__

