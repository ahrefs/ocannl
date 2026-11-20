/* gh-ocannl-1638: the Metal device's GPU core count, which Metal itself does not publish.

   An Apple-silicon GPU's IORegistry entry -- the AGX accelerator service whose registry ID is the
   one [MTLDevice.registryID] reports -- carries a "gpu-core-count" number property (what
   `ioreg -rc AGXAccelerator -d1` prints). The launch fill's multiprocessor count reads it. */

#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>

#include <caml/alloc.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>

/* The "gpu-core-count" property of the registry entry with the given ID, or 0 where there is no
   such entry or it carries no positive count (a non-AGX GPU). MACH_PORT_NULL is the default main
   port on every macOS release, where the kIOMainPortDefault spelling needs macOS 12. */
static long gpu_core_count_of_registry_id(uint64_t registry_id) {
  long count = 0;
  io_service_t service =
      IOServiceGetMatchingService(MACH_PORT_NULL, IORegistryEntryIDMatching(registry_id));
  if (service == IO_OBJECT_NULL) return 0;
  CFTypeRef property = IORegistryEntryCreateCFProperty(service, CFSTR("gpu-core-count"),
                                                       kCFAllocatorDefault, 0);
  if (property != NULL) {
    if (CFGetTypeID(property) != CFNumberGetTypeID() ||
        !CFNumberGetValue((CFNumberRef)property, kCFNumberLongType, &count))
      count = 0;
    CFRelease(property);
  }
  IOObjectRelease(service);
  return count > 0 ? count : 0;
}

CAMLprim value ocannl_metal_gpu_core_count(value v_registry_id) {
  CAMLparam1(v_registry_id);
  CAMLreturn(Val_long(gpu_core_count_of_registry_id((uint64_t)Int64_val(v_registry_id))));
}

/* Read the top-level ProductBuildVersion only when it is a plist string. */
CAMLprim value ocannl_metal_product_build_version(value v_contents) {
  CAMLparam1(v_contents);
  CAMLlocal1(build);
  CFDataRef data = CFDataCreate(kCFAllocatorDefault, (const UInt8 *)String_val(v_contents),
                               caml_string_length(v_contents));
  if (data == NULL) CAMLreturn(Val_none);
  CFPropertyListRef plist = CFPropertyListCreateWithData(
      kCFAllocatorDefault, data, kCFPropertyListImmutable, NULL, NULL);
  CFRelease(data);
  if (plist == NULL) CAMLreturn(Val_none);

  CFStringRef version = NULL;
  if (CFGetTypeID(plist) == CFDictionaryGetTypeID()) {
    CFTypeRef entry = CFDictionaryGetValue((CFDictionaryRef)plist, CFSTR("ProductBuildVersion"));
    if (entry != NULL && CFGetTypeID(entry) == CFStringGetTypeID()) version = (CFStringRef)entry;
  }
  if (version == NULL) {
    CFRelease(plist);
    CAMLreturn(Val_none);
  }

  CFRange range = CFRangeMake(0, CFStringGetLength(version));
  CFIndex len = 0;
  if (CFStringGetBytes(version, range, kCFStringEncodingUTF8, 0, false, NULL, 0, &len) !=
      range.length) {
    CFRelease(plist);
    CAMLreturn(Val_none);
  }
  build = caml_alloc_string(len);
  CFStringGetBytes(version, range, kCFStringEncodingUTF8, 0, false, Bytes_val(build), len, NULL);
  CFRelease(plist);
  CAMLreturn(caml_alloc_some(build));
}
