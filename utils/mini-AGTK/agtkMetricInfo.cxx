#include "agtkMetricInfo.h"

namespace agtk
{
//----------------------------------------------------------------------------
MetricInfo::MetricInfo(
  double value,
  const char* name,
  const char* longName,
  MetricUnits units /*= MetricUnits::None*/,
  const char* description /*= ""*/,
  const char* formula /*= ""*/)
  : m_Value(value)
  , m_Name(name)
  , m_LongName(longName)
  , m_Units(units)
  , m_Description(description)
  , m_Formula(formula)
{
}
}
