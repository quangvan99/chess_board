/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef __GST_NVOSDPADDING_H__
#define __GST_NVOSDPADDING_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideofilter.h>
#include <stdlib.h>
#include <nvll_osd_api.h>
#include <gstnvdsmeta.h>
// #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc/types_c.h>
#define MAX_BG_CLR 20

G_BEGIN_DECLS
/* Standard GStreamer boilerplate */
#define GST_TYPE_NVOSDPADDING \
  (GST_NVOSDPADDING_get_type())
#define GST_NVOSDPADDING(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_NVOSDPADDING,GstNvOsdPadding))
#define GST_NVOSDPADDING_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_NVOSDPADDING,GstNvOsdPaddingClass))
#define GST_IS_NVOSDPADDING(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_NVOSDPADDING))
#define GST_IS_NVOSDPADDING_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_NVOSDPADDING))
/* Version number of package */
#define VERSION "1.8.2"
#define PACKAGE_DESCRIPTION "Gstreamer NVIDIA plugin for video padding"
/* Define under which licence the package has been released */
#define PACKAGE_LICENSE "Proprietary"
#define PACKAGE_NAME "GStreamer NVIDIA Padding Plugin"
/* Define to the home page for this package. */
#define PACKAGE_URL "http://nvidia.com/"
#define GST_CAPS_FEATURE_MEMORY_NVMM      "memory:NVMM"
typedef struct _GstNvOsdPadding GstNvOsdPadding;
typedef struct _GstNvOsdPaddingClass GstNvOsdPaddingClass;

// Thêm struct cho text position
struct TextPosition {
    int x;
    int y;
};
/**
 * GstNvOsdPadding element structure.
 */
struct _GstNvOsdPadding
{
  /** Should be the first member when extending from GstBaseTransform. */
  GstBaseTransform parent_instance;

  /* Width of buffer. */
  gint width;
  /* Height of buffer. */
  gint height;

  /** Pointer to the nvosdpadding context. */
  void *nvosdpadding_context;
  /** Enum indicating how the objects are drawn,
      i.e., CPU, GPU or VIC (for Jetson only). */
  NvOSD_Mode nvosdpadding_mode;

  /** Boolean value indicating whether clock is enabled. */
  gboolean show_clock;
  /** Structure containing text params for clock. */
  NvOSD_TextParams clock_text_params;

  /** List of strings to be drawn. */
  NvOSD_TextParams *text_params;
  /** List of rectangles to be drawn. */
  NvOSD_RectParams *rect_params;
  /** List of rectangles for segment masks to be drawn. */
  NvOSD_RectParams *mask_rect_params;
  /** List of segment masks to be drawn. */
  NvOSD_MaskParams *mask_params;
  /** List of lines to be drawn. */
  NvOSD_LineParams *line_params;
  /** List of arrows to be drawn. */
  NvOSD_ArrowParams *arrow_params;
  /** List of circles to be drawn. */
  NvOSD_CircleParams *circle_params;

  /** Number of rectangles to be drawn for a frame. */
  guint num_rect;
  /** Number of segment masks to be drawn for a frame. */
  guint num_segments;
  /** Number of strings to be drawn for a frame. */
  guint num_strings;
  /** Number of lines to be drawn for a frame. */
  guint num_lines;
  /** Number of arrows to be drawn for a frame. */
  guint num_arrows;
  /** Number of circles to be drawn for a frame. */
  guint num_circles;
  /** Size padding */
  guint padding_size;

  /** Structure containing details of rectangles to be drawn for a frame. */
  NvOSD_FrameRectParams *frame_rect_params;
  /** Structure containing details of segment masks to be drawn for a frame. */
  NvOSD_FrameSegmentMaskParams *frame_mask_params;
  /** Structure containing details of text to be overlayed for a frame. */
  NvOSD_FrameTextParams *frame_text_params;
  /** Structure containing details of lines to be drawn for a frame. */
  NvOSD_FrameLineParams *frame_line_params;
  /** Structure containing details of arrows to be drawn for a frame. */
  NvOSD_FrameArrowParams *frame_arrow_params;
  /** Structure containing details of circles to be drawn for a frame. */
  NvOSD_FrameCircleParams *frame_circle_params;

  /** Font of the text to be displayed. */
  gchar *font;
  /** Color of the clock, if enabled. */
  guint clock_color;
  /** Font size of the clock, if enabled. */
  guint clock_font_size;
  /** Border width of object. */
  guint border_width;
  /** Integer indicating the frame number. */
  guint frame_num;
  /** Boolean indicating whether text is to be drawn. */
  gboolean draw_text;
  /** Boolean indicating whether bounding is to be drawn. */
  gboolean draw_bbox;
  /** Boolean indicating whether instance mask is to be drawn. */
  gboolean draw_mask;

  /**Array containing color info for blending */
  NvOSD_Color_info color_info[MAX_BG_CLR];
  gboolean enable_padding;
  guint padding_color[4];
  gchar *padding_text;     // Add this line for padding text
  gchar *num_sources;
  TextPosition text_position;
  /** Integer indicating number of detected classes. */
  int num_class_entries;
  /** Integer indicating gpu id to be used. */
  guint gpu_id;
  /** Pointer to the converted buffer. */
  void *conv_buf;

};


/* GStreamer boilerplate. */
struct _GstNvOsdPaddingClass
{
  GstBaseTransformClass parent_class;
};

GType GST_NVOSDPADDING_get_type (void);

G_END_DECLS
#endif /* __GST_NVOSDPADDING_H__ */
