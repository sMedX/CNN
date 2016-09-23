
#ifndef AGTKCore_EXPORT_H
#define AGTKCore_EXPORT_H

#ifdef AGTK_STATIC
#  define AGTKCore_EXPORT
#  define AGTKCore_HIDDEN
#else
#  ifndef AGTKCore_EXPORT
#    ifdef AGTKCore_EXPORTS
        /* We are building this library */
#      define AGTKCore_EXPORT 
#    else
        /* We are using this library */
#      define AGTKCore_EXPORT 
#    endif
#  endif

#  ifndef AGTKCore_HIDDEN
#    define AGTKCore_HIDDEN 
#  endif
#endif

#ifndef AGTKCORE_DEPRECATED
#  define AGTKCORE_DEPRECATED __declspec(deprecated)
#endif

#ifndef AGTKCORE_DEPRECATED_EXPORT
#  define AGTKCORE_DEPRECATED_EXPORT AGTKCore_EXPORT AGTKCORE_DEPRECATED
#endif

#ifndef AGTKCORE_DEPRECATED_NO_EXPORT
#  define AGTKCORE_DEPRECATED_NO_EXPORT AGTKCore_HIDDEN AGTKCORE_DEPRECATED
#endif

#define DEFINE_NO_DEPRECATED 0
#if DEFINE_NO_DEPRECATED
# define AGTKCORE_NO_DEPRECATED
#endif

#endif
