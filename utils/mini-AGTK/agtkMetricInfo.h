#ifndef __agtkMetricInfo_h
#define __agtkMetricInfo_h

#include <vector>

#include "agtkExport.h"

namespace agtk
{
enum MetricUnits
{
  None,
  Millimeters
};

class AGTK_EXPORT MetricInfo
{
public:
  MetricInfo(
    double value,
    const char* name,
    const char* longName,
    MetricUnits units = MetricUnits::None,
    const char* description = "",
    const char* formula = "");

  double getValue() const { return m_Value; }
  const char* getName() const { return m_Name; }
  const char* getLongName() const { return m_LongName; }
  MetricUnits getUnits() const { return m_Units; }
  const char* getDescription() const { return m_Description; }
  const char* getFormula() const { return m_Formula; }

private:
  double m_Value;
  const char* m_Name;
  const char* m_LongName;
  MetricUnits m_Units;
  const char* m_Description;
  const char* m_Formula;
};

typedef std::vector<MetricInfo> MetricsInfo;
}

#endif // __agtkMetricInfo_h
