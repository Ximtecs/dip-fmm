// SPDX-License-Identifier: Apache-2.0
#pragma once

// The cache tests give each process a private cache directory through the
// POSIX environment calls. Windows has getpid in <process.h> but no setenv or
// unsetenv, so they are supplied here on top of the CRT's _putenv_s.
#ifdef _WIN32
#include <process.h>
#include <stdlib.h>

#include <string>

inline int setenv(const char* name, const char* value, int /*overwrite*/) {
  return ::_putenv_s(name, value);
}

// std::filesystem::path::c_str() is a wide string on Windows.
inline int setenv(const char* name, const wchar_t* value, int /*overwrite*/) {
  const std::wstring wide_name(name, name + std::char_traits<char>::length(name));
  return ::_wputenv_s(wide_name.c_str(), value);
}

// An empty value removes the variable from the CRT environment.
inline int unsetenv(const char* name) { return ::_putenv_s(name, ""); }
#else
#include <stdlib.h>
#include <unistd.h>
#endif
