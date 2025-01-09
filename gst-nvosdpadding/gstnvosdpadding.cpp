/*
 * SPDX-FileCopyrightText: Copyright (c) 2016-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <stdio.h>
#include <gst/gst.h>

#include <gst/video/video.h>
#include <gst/base/gstbasetransform.h>
// #include "GstNvOsdPadding.h"
#include "gstnvosdpadding.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/types_c.h>

#include <nvbufsurface.h>
#include <nvtx3/nvToolsExt.h>
#include <gst-nvdscustomevent.h>
GST_DEBUG_CATEGORY_STATIC (GST_NVOSDPADDING_debug);
#define GST_CAT_DEFAULT GST_NVOSDPADDING_debug

/* For hw blending, color should be of the form:
   class_id1, R, G, B, A:class_id2, R, G, B, A */
#define DEFAULT_CLR "0,0.0,1.0,0.0,0.3:1,0.0,1.0,1.0,0.3:2,0.0,0.0,1.0,0.3:3,1.0,1.0,0.0,0.3"
#define MAX_OSD_ELEMS 1024

/* Filter signals and args */
enum
{
  /* FILL ME */
  LAST_SIGNAL
};

/* Enum to identify properties */
enum
{
  PROP_0,
  PROP_PADDING_SIZE,
  PROP_ENABLE_PADDING,
  PROP_PADDING_COLOR,
  PROP_SHOW_CLOCK,
  PROP_SHOW_TEXT,
  PROP_SHOW_BBOX,
  PROP_SHOW_MASK,
  PROP_CLOCK_FONT,
  PROP_CLOCK_FONT_SIZE,
  PROP_CLOCK_X_OFFSET,
  PROP_CLOCK_Y_OFFSET,
  PROP_CLOCK_COLOR,
  PROP_PROCESS_MODE,
  PROP_GPU_DEVICE_ID
};

/* the capabilities of the inputs and outputs. */
static GstStaticPadTemplate sink_factory =
GST_STATIC_PAD_TEMPLATE (
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

static GstStaticPadTemplate src_factory =
GST_STATIC_PAD_TEMPLATE (
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE_WITH_FEATURES
        (GST_CAPS_FEATURE_MEMORY_NVMM,
            "{ NV12, RGBA }")));

/* Default values for properties */
#define DEFAULT_FONT_SIZE 12
#define DEFAULT_FONT "Serif"
#define GST_NV_OSD_DEFAULT_PROCESS_MODE MODE_GPU
#define MAX_FONT_SIZE 60
#define DEFAULT_BORDER_WIDTH 4
#define DEFAULT_PADDING_SIZE 10
#define DEFAULT_ENABLE_PADDING FALSE
#define DEFAULT_PADDING_COLOR_R 255
#define DEFAULT_PADDING_COLOR_G 255
#define DEFAULT_PADDING_COLOR_B 255
#define DEFAULT_PADDING_COLOR_A 255

/* Define our element type. Standard GObject/GStreamer boilerplate stuff */
#define GST_NVOSDPADDING_parent_class parent_class
G_DEFINE_TYPE (GstNvOsdPadding, GST_NVOSDPADDING, GST_TYPE_BASE_TRANSFORM);

#define GST_TYPE_NV_OSD_PROCESS_MODE (GST_NVOSDPADDING_process_mode_get_type ())

static GQuark _dsmeta_quark;

static GType
GST_NVOSDPADDING_process_mode_get_type (void)
{
  static GType qtype = 0;

  if (qtype == 0) {
    static const GEnumValue values[] = {
      {MODE_CPU, "CPU_MODE", "MODE_CPU"},
      {MODE_GPU, "GPU_MODE", "MODE_GPU"},
#ifdef PLATFORM_TEGRA
      {MODE_NONE,
            "Invalid mode. Falls back to GPU",
          "MODE_NONE"},
#endif
      {0, NULL, NULL}
    };

    qtype = g_enum_register_static ("GstNvOsdPaddingMode", values);
  }
  return qtype;
}

static void GST_NVOSDPADDING_finalize (GObject * object);
static void GST_NVOSDPADDING_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void GST_NVOSDPADDING_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);
static GstFlowReturn GST_NVOSDPADDING_transform_ip (GstBaseTransform * trans,
    GstBuffer * buf);
static gboolean GST_NVOSDPADDING_start (GstBaseTransform * btrans);
static gboolean GST_NVOSDPADDING_stop (GstBaseTransform * btrans);
static gboolean GST_NVOSDPADDING_parse_color (GstNvOsdPadding * nvosdpadding,
    guint clock_color);
static gboolean GST_NVOSDPADDING_sink_event (GstBaseTransform * trans,
    GstEvent * event);

static gboolean
GST_NVOSDPADDING_sink_event (GstBaseTransform * trans, GstEvent * event)
{
  GstNvOsdPadding *nvosdpadding = GST_NVOSDPADDING (trans);

  if ((GstNvDsCustomEventType) GST_EVENT_TYPE (event) ==
      GST_NVEVENT_OSD_PROCESS_MODE_UPDATE) {
    gchar *stream_id = NULL;
    guint process_mode = 0;

    gst_nvevent_parse_osd_process_mode_update (event, &stream_id,
        &process_mode);

    nvosdpadding->nvosdpadding_mode = process_mode == 0 ? MODE_CPU : MODE_GPU;

    int flag_integrated = -1;
    cudaDeviceGetAttribute (&flag_integrated, cudaDevAttrIntegrated,
        nvosdpadding->gpu_id);
    if (!flag_integrated && nvosdpadding->nvosdpadding_mode > MODE_GPU) {
      g_print ("WARN !! Invalid mode selected, Falling back to GPU\n");
      nvosdpadding->nvosdpadding_mode = MODE_GPU;
    }
  }

  return GST_BASE_TRANSFORM_CLASS (parent_class)->sink_event (trans, event);
}

static GstCaps *
GST_NVOSDPADDING_transform_caps (GstBaseTransform * trans,
    GstPadDirection direction, GstCaps * caps, GstCaps * filter)
{
  GstNvOsdPadding *nvosdpadding = GST_NVOSDPADDING (trans);
  GstCaps *ret;
  GstCaps *new_caps;
  GstCaps *caps_rgba =
      gst_caps_from_string ("video/x-raw(memory:NVMM), format=(string)RGBA");

  GST_DEBUG_OBJECT (trans, "identity from: %" GST_PTR_FORMAT, caps);
  if (filter) {
    ret = gst_caps_intersect_full (filter, caps, GST_CAPS_INTERSECT_FIRST);
  } else {
    ret = gst_caps_ref (caps);
  }

  /* Force to RGBA format for CPU mode. */
  if (nvosdpadding->nvosdpadding_mode == MODE_CPU) {
    new_caps = gst_caps_intersect_full (ret, caps_rgba, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref (ret);
    ret = new_caps;
  }
  gst_caps_unref (caps_rgba);

  return ret;
}

/**
 * Called when source / sink pad capabilities have been negotiated.
 */
static gboolean
GST_NVOSDPADDING_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  gboolean ret = TRUE;

  GstNvOsdPadding *nvosdpadding = GST_NVOSDPADDING (trans);
  gint width = 0, height = 0;
  cudaError_t CUerr = cudaSuccess;

  nvosdpadding->frame_num = 0;

  GstStructure *structure = gst_caps_get_structure (incaps, 0);

  GST_OBJECT_LOCK (nvosdpadding);
  if (!gst_structure_get_int (structure, "width", &width) ||
      !gst_structure_get_int (structure, "height", &height)) {
    // GST_ELEMENT_ERROR (nvosdpadding, STREAM, FAILED,
    //     ("caps without width/height"), NULL);
    ret = FALSE;
    goto exit_set_caps;
  }
  if (nvosdpadding->nvosdpadding_context && nvosdpadding->width == width
      && nvosdpadding->height == height) {
    goto exit_set_caps;
  }

  CUerr = cudaSetDevice (nvosdpadding->gpu_id);
  if (CUerr != cudaSuccess) {
    ret = FALSE;
    // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
    //     ("Unable to set device"), NULL);
    goto exit_set_caps;
  }

  nvosdpadding->width = width;
  nvosdpadding->height = height;

  if (nvosdpadding->show_clock)
    nvll_osd_set_clock_params (nvosdpadding->nvosdpadding_context,
        &nvosdpadding->clock_text_params);

  nvosdpadding->conv_buf =
      nvll_osd_set_params (nvosdpadding->nvosdpadding_context, nvosdpadding->width,
      nvosdpadding->height);

exit_set_caps:
  GST_OBJECT_UNLOCK (nvosdpadding);
  return ret;
}

/**
 * Initialize all resources.
 */
static gboolean
GST_NVOSDPADDING_start (GstBaseTransform * btrans)
{
  GstNvOsdPadding *nvosdpadding = GST_NVOSDPADDING (btrans);

  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaSetDevice (nvosdpadding->gpu_id);
  if (CUerr != cudaSuccess) {
    // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
    //     ("Unable to set device"), NULL);
    return FALSE;
  }
  GST_LOG_OBJECT (nvosdpadding, "SETTING CUDA DEVICE = %d in nvosdpadding func=%s\n",
      nvosdpadding->gpu_id, __func__);

  nvosdpadding->nvosdpadding_context = nvll_osd_create_context ();

  if (nvosdpadding->nvosdpadding_context == NULL) {
    // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
    //     ("Unable to create context nvosdpadding"), NULL);
    return FALSE;
  }

  int flag_integrated = -1;
  cudaDeviceGetAttribute (&flag_integrated, cudaDevAttrIntegrated,
      nvosdpadding->gpu_id);
  if (!flag_integrated && nvosdpadding->nvosdpadding_mode > MODE_GPU) {
    g_print ("WARN !! Invalid mode selected, Falling back to GPU\n");
    nvosdpadding->nvosdpadding_mode = MODE_GPU;
  }

  if (nvosdpadding->show_clock) {
    nvll_osd_set_clock_params (nvosdpadding->nvosdpadding_context,
        &nvosdpadding->clock_text_params);
  }

  return TRUE;
}

/**
 * Free up all the resources
 */
static gboolean
GST_NVOSDPADDING_stop (GstBaseTransform * btrans)
{
  GstNvOsdPadding *nvosdpadding = GST_NVOSDPADDING (btrans);

  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaSetDevice (nvosdpadding->gpu_id);
  if (CUerr != cudaSuccess) {
    // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
    //     ("Unable to set device"), NULL);
    return FALSE;
  }
  GST_LOG_OBJECT (nvosdpadding, "SETTING CUDA DEVICE = %d in nvosdpadding func=%s\n",
      nvosdpadding->gpu_id, __func__);

  if (nvosdpadding->nvosdpadding_context)
    nvll_osd_destroy_context (nvosdpadding->nvosdpadding_context);

  nvosdpadding->nvosdpadding_context = NULL;
  nvosdpadding->width = 0;
  nvosdpadding->height = 0;

  return TRUE;
}

int frame_num = 0;

/**
 * Called when element recieves an input buffer from upstream element.
 */
static GstFlowReturn
GST_NVOSDPADDING_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstNvOsdPadding *nvosdpadding = GST_NVOSDPADDING (trans);
  GstMapInfo inmap = GST_MAP_INFO_INIT;
  unsigned int rect_cnt = 0;
  unsigned int segment_cnt = 0;
  unsigned int text_cnt = 0;
  unsigned int line_cnt = 0;
  unsigned int arrow_cnt = 0;
  unsigned int circle_cnt = 0;
  unsigned int i = 0;

  gpointer state = NULL;
  NvBufSurface *surface = NULL;
  NvDsBatchMeta *batch_meta = NULL;

  if (!gst_buffer_map (buf, &inmap, GST_MAP_READ)) {
    // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
    //     ("Unable to map info from buffer"), NULL);
    return GST_FLOW_ERROR;
  }

  nvds_set_input_system_timestamp (buf, GST_ELEMENT_NAME (nvosdpadding));

  cudaError_t CUerr = cudaSuccess;
  CUerr = cudaSetDevice (nvosdpadding->gpu_id);
  if (CUerr != cudaSuccess) {
    // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
    //     ("Unable to set device"), NULL);
    return GST_FLOW_ERROR;
  }
  GST_LOG_OBJECT (nvosdpadding, "SETTING CUDA DEVICE = %d in nvosdpadding func=%s\n",
      nvosdpadding->gpu_id, __func__);

  surface = (NvBufSurface *) inmap.data;

  /* Get metadata. Update rectangle and text params */
  GstMeta *gst_meta;
  NvDsMeta *dsmeta;
  char context_name[100];
  snprintf (context_name, sizeof (context_name), "%s_(Frame=%u)",
      GST_ELEMENT_NAME (nvosdpadding), nvosdpadding->frame_num);
  nvtxRangePushA (context_name);
  while ((gst_meta = gst_buffer_iterate_meta (buf, &state))) {
    if (gst_meta_api_type_has_tag (gst_meta->info->api, _dsmeta_quark)) {
      dsmeta = (NvDsMeta *) gst_meta;
      if (dsmeta->meta_type == NVDS_BATCH_GST_META) {
        batch_meta = (NvDsBatchMeta *) dsmeta->meta_data;
        break;
      }
    }
  }

  NvDsMetaList *l = NULL;
  NvDsMetaList *full_obj_meta_list = NULL;
  if (batch_meta)
    full_obj_meta_list = batch_meta->obj_meta_pool->full_list;
  NvDsObjectMeta *object_meta = NULL;

  for (l = full_obj_meta_list; l != NULL; l = l->next) {
    object_meta = (NvDsObjectMeta *) (l->data);
    if (nvosdpadding->draw_bbox) {
      nvosdpadding->rect_params[rect_cnt] = object_meta->rect_params;
      rect_cnt++;
    }
    // printf("*******************************************test");
    // printf("enable_padding: %d, padding_size: %d\n", nvosdpadding->enable_padding, nvosdpadding->padding_size);
    if (nvosdpadding->enable_padding && nvosdpadding->padding_size > 0) {
        printf("*******************************************test\n");
      unsigned char *src_data = new unsigned char[surface->surfaceList[i].dataSize];

      cudaMemcpy(src_data, surface->surfaceList[i].dataPtr, surface->surfaceList[i].dataSize, cudaMemcpyDeviceToHost);
      int frame_width = surface->surfaceList[i].width;
      int frame_height = surface->surfaceList[i].height;
      size_t frame_step = surface->surfaceList[i].pitch;

      cv::Mat rgba(frame_height, frame_width, CV_8UC4, src_data, frame_step);
      cv::imwrite("./output/original_frame_padding.png", rgba);
    }


    if (rect_cnt == MAX_OSD_ELEMS) {
      nvosdpadding->frame_rect_params->num_rects = rect_cnt;
      nvosdpadding->frame_rect_params->rect_params_list = nvosdpadding->rect_params;
      /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_rect_params->surf' instead */
      nvosdpadding->frame_rect_params->buf_ptr = NULL;
      nvosdpadding->frame_rect_params->mode = nvosdpadding->nvosdpadding_mode;
      nvosdpadding->frame_rect_params->surf = surface;
      if (nvll_osd_draw_rectangles (nvosdpadding->nvosdpadding_context,
              nvosdpadding->frame_rect_params) == -1) {
        // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
        //     ("Unable to draw rectangles"), NULL);
        return GST_FLOW_ERROR;
      }
      rect_cnt = 0;
    }
    if (nvosdpadding->draw_mask && object_meta->mask_params.data &&
        object_meta->mask_params.size > 0) {
      nvosdpadding->mask_rect_params[segment_cnt] = object_meta->rect_params;
      nvosdpadding->mask_params[segment_cnt++] = object_meta->mask_params;
      if (segment_cnt == MAX_OSD_ELEMS) {
        nvosdpadding->frame_mask_params->num_segments = segment_cnt;
        nvosdpadding->frame_mask_params->rect_params_list =
            nvosdpadding->mask_rect_params;
        nvosdpadding->frame_mask_params->mask_params_list = nvosdpadding->mask_params;
        /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_mask_params->surf' instead */
        nvosdpadding->frame_mask_params->buf_ptr = NULL;
        nvosdpadding->frame_mask_params->mode = nvosdpadding->nvosdpadding_mode;
        nvosdpadding->frame_mask_params->surf = surface;
        if (nvll_osd_draw_segment_masks (nvosdpadding->nvosdpadding_context,
                nvosdpadding->frame_mask_params) == -1) {
          // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
          //     ("Unable to draw rectangles"), NULL);
          return GST_FLOW_ERROR;
        }
        segment_cnt = 0;
      }
    }
    if (object_meta->text_params.display_text)
      nvosdpadding->text_params[text_cnt++] = object_meta->text_params;
    if (text_cnt == MAX_OSD_ELEMS) {
      nvosdpadding->frame_text_params->num_strings = text_cnt;
      nvosdpadding->frame_text_params->text_params_list = nvosdpadding->text_params;
      /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_text_params->surf' instead */
      nvosdpadding->frame_text_params->buf_ptr = NULL;
      nvosdpadding->frame_text_params->mode = nvosdpadding->nvosdpadding_mode;
      nvosdpadding->frame_rect_params->surf = surface;
      if (nvll_osd_put_text (nvosdpadding->nvosdpadding_context,
              nvosdpadding->frame_text_params) == -1) {
        // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
        //     ("Unable to draw text"), NULL);
        return GST_FLOW_ERROR;
      }
      text_cnt = 0;
    }
  }

  NvDsMetaList *display_meta_list = NULL;
  if (batch_meta)
    display_meta_list = batch_meta->display_meta_pool->full_list;
  NvDsDisplayMeta *display_meta = NULL;

  /* Get objects to be drawn from display meta.
   * Draw objects if count equals MAX_OSD_ELEMS.
   */
  for (l = display_meta_list; l != NULL; l = l->next) {
    display_meta = (NvDsDisplayMeta *) (l->data);

    unsigned int cnt = 0;
    for (cnt = 0; cnt < display_meta->num_rects; cnt++) {
      nvosdpadding->rect_params[rect_cnt++] = display_meta->rect_params[cnt];
      if (rect_cnt == MAX_OSD_ELEMS) {
        nvosdpadding->frame_rect_params->num_rects = rect_cnt;
        nvosdpadding->frame_rect_params->rect_params_list = nvosdpadding->rect_params;
        /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_rect_params->surf' instead */
        nvosdpadding->frame_rect_params->buf_ptr = NULL;
        nvosdpadding->frame_rect_params->mode = nvosdpadding->nvosdpadding_mode;
        nvosdpadding->frame_rect_params->surf = surface;
        if (nvll_osd_draw_rectangles (nvosdpadding->nvosdpadding_context,
                nvosdpadding->frame_rect_params) == -1) {
          // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
          //     ("Unable to draw rectangles"), NULL);
          return GST_FLOW_ERROR;
        }
        rect_cnt = 0;
      }
    }

    for (cnt = 0; cnt < display_meta->num_labels; cnt++) {
      if (display_meta->text_params[cnt].display_text) {
        nvosdpadding->text_params[text_cnt++] = display_meta->text_params[cnt];
        if (text_cnt == MAX_OSD_ELEMS) {
          nvosdpadding->frame_text_params->num_strings = text_cnt;
          nvosdpadding->frame_text_params->text_params_list = nvosdpadding->text_params;
          /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_text_params->surf' instead */
          nvosdpadding->frame_text_params->buf_ptr = NULL;
          nvosdpadding->frame_text_params->mode = nvosdpadding->nvosdpadding_mode;
          nvosdpadding->frame_text_params->surf = surface;
          if (nvll_osd_put_text (nvosdpadding->nvosdpadding_context,
                  nvosdpadding->frame_text_params) == -1) {
            // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
            //     ("Unable to draw text"), NULL);
            return GST_FLOW_ERROR;
          }
          text_cnt = 0;
        }
      }
    }

    for (cnt = 0; cnt < display_meta->num_lines; cnt++) {
      nvosdpadding->line_params[line_cnt++] = display_meta->line_params[cnt];
      if (line_cnt == MAX_OSD_ELEMS) {
        nvosdpadding->frame_line_params->num_lines = line_cnt;
        nvosdpadding->frame_line_params->line_params_list = nvosdpadding->line_params;
        /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_line_params->surf' instead */
        nvosdpadding->frame_line_params->buf_ptr = NULL;
        nvosdpadding->frame_line_params->mode = nvosdpadding->nvosdpadding_mode;
        nvosdpadding->frame_line_params->surf = surface;
        if (nvll_osd_draw_lines (nvosdpadding->nvosdpadding_context,
                nvosdpadding->frame_line_params) == -1) {
          // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
          //     ("Unable to draw lines"), NULL);
          return GST_FLOW_ERROR;
        }
        line_cnt = 0;
      }
    }

    for (cnt = 0; cnt < display_meta->num_arrows; cnt++) {
      nvosdpadding->arrow_params[arrow_cnt++] = display_meta->arrow_params[cnt];
      if (arrow_cnt == MAX_OSD_ELEMS) {
        nvosdpadding->frame_arrow_params->num_arrows = arrow_cnt;
        nvosdpadding->frame_arrow_params->arrow_params_list = nvosdpadding->arrow_params;
        /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_arrow_params->surf' instead */
        nvosdpadding->frame_arrow_params->buf_ptr = NULL;
        nvosdpadding->frame_arrow_params->mode = nvosdpadding->nvosdpadding_mode;
        nvosdpadding->frame_arrow_params->surf = surface;
        if (nvll_osd_draw_arrows (nvosdpadding->nvosdpadding_context,
                nvosdpadding->frame_arrow_params) == -1) {
          // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
          //     ("Unable to draw arrows"), NULL);
          return GST_FLOW_ERROR;
        }
        arrow_cnt = 0;
      }
    }

    for (cnt = 0; cnt < display_meta->num_circles; cnt++) {
      nvosdpadding->circle_params[circle_cnt++] = display_meta->circle_params[cnt];
      if (circle_cnt == MAX_OSD_ELEMS) {
        nvosdpadding->frame_circle_params->num_circles = circle_cnt;
        nvosdpadding->frame_circle_params->circle_params_list =
            nvosdpadding->circle_params;
        /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_circle_params->surf' instead */
        nvosdpadding->frame_circle_params->buf_ptr = NULL;
        nvosdpadding->frame_circle_params->mode = nvosdpadding->nvosdpadding_mode;
        nvosdpadding->frame_circle_params->surf = surface;
        if (nvll_osd_draw_circles (nvosdpadding->nvosdpadding_context,
                nvosdpadding->frame_circle_params) == -1) {
          // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
          //     ("Unable to draw circles"), NULL);
          return GST_FLOW_ERROR;
        }
        circle_cnt = 0;
      }
    }
    i++;
  }

  nvosdpadding->num_rect = rect_cnt;
  nvosdpadding->num_segments = segment_cnt;
  nvosdpadding->num_strings = text_cnt;
  nvosdpadding->num_lines = line_cnt;
  nvosdpadding->num_arrows = arrow_cnt;
  nvosdpadding->num_circles = circle_cnt;
  if (rect_cnt != 0 && nvosdpadding->draw_bbox) {
    nvosdpadding->frame_rect_params->num_rects = nvosdpadding->num_rect;
    nvosdpadding->frame_rect_params->rect_params_list = nvosdpadding->rect_params;
    /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_rect_params->surf' instead */
    nvosdpadding->frame_rect_params->buf_ptr = NULL;
    nvosdpadding->frame_rect_params->mode = nvosdpadding->nvosdpadding_mode;
    nvosdpadding->frame_rect_params->surf = surface;
    if (nvll_osd_draw_rectangles (nvosdpadding->nvosdpadding_context,
            nvosdpadding->frame_rect_params) == -1) {
      // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
      //     ("Unable to draw rectangles"), NULL);
      return GST_FLOW_ERROR;
    }
  }

  if (segment_cnt != 0 && nvosdpadding->draw_mask) {
    nvosdpadding->frame_mask_params->num_segments = nvosdpadding->num_segments;
    nvosdpadding->frame_mask_params->rect_params_list = nvosdpadding->mask_rect_params;
    nvosdpadding->frame_mask_params->mask_params_list = nvosdpadding->mask_params;
    /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_mask_params->surf' instead */
    nvosdpadding->frame_mask_params->buf_ptr = NULL;
    nvosdpadding->frame_mask_params->mode = nvosdpadding->nvosdpadding_mode;
    nvosdpadding->frame_mask_params->surf = surface;
    if (nvll_osd_draw_segment_masks (nvosdpadding->nvosdpadding_context,
            nvosdpadding->frame_mask_params) == -1) {
      // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
      //     ("Unable to draw segment masks"), NULL);
      return GST_FLOW_ERROR;
    }
  }

  if ((nvosdpadding->show_clock || text_cnt) && nvosdpadding->draw_text) {
    nvosdpadding->frame_text_params->num_strings = nvosdpadding->num_strings;
    nvosdpadding->frame_text_params->text_params_list = nvosdpadding->text_params;
    /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_text_params->surf' instead */
    nvosdpadding->frame_text_params->buf_ptr = NULL;
    nvosdpadding->frame_text_params->mode = nvosdpadding->nvosdpadding_mode;
    nvosdpadding->frame_text_params->surf = surface;
    if (nvll_osd_put_text (nvosdpadding->nvosdpadding_context,
            nvosdpadding->frame_text_params) == -1) {
      // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED, ("Unable to draw text"),
      //     NULL);
      return GST_FLOW_ERROR;
    }
  }

  if (line_cnt != 0) {
    nvosdpadding->frame_line_params->num_lines = nvosdpadding->num_lines;
    nvosdpadding->frame_line_params->line_params_list = nvosdpadding->line_params;
    /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_line_params->surf' instead */
    nvosdpadding->frame_line_params->buf_ptr = NULL;
    nvosdpadding->frame_line_params->mode = nvosdpadding->nvosdpadding_mode;
    nvosdpadding->frame_line_params->surf = surface;
    if (nvll_osd_draw_lines (nvosdpadding->nvosdpadding_context,
            nvosdpadding->frame_line_params) == -1) {
      // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED, ("Unable to draw lines"),
      //     NULL);
      return GST_FLOW_ERROR;
    }
  }

  if (arrow_cnt != 0) {
    nvosdpadding->frame_arrow_params->num_arrows = nvosdpadding->num_arrows;
    nvosdpadding->frame_arrow_params->arrow_params_list = nvosdpadding->arrow_params;
    /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_arrow_params->surf' instead */
    nvosdpadding->frame_arrow_params->buf_ptr = NULL;
    nvosdpadding->frame_arrow_params->mode = nvosdpadding->nvosdpadding_mode;
    nvosdpadding->frame_arrow_params->surf = surface;
    if (nvll_osd_draw_arrows (nvosdpadding->nvosdpadding_context,
            nvosdpadding->frame_arrow_params) == -1) {
      // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
      //     ("Unable to draw arrows"), NULL);
      return GST_FLOW_ERROR;
    }
  }

  if (circle_cnt != 0) {
    nvosdpadding->frame_circle_params->num_circles = nvosdpadding->num_circles;
    nvosdpadding->frame_circle_params->circle_params_list = nvosdpadding->circle_params;
    /** Use of buf_ptr is deprecated, use 'nvosdpadding->frame_circle_params->surf' instead */
    nvosdpadding->frame_circle_params->buf_ptr = NULL;
    nvosdpadding->frame_circle_params->mode = nvosdpadding->nvosdpadding_mode;
    nvosdpadding->frame_circle_params->surf = surface;
    if (nvll_osd_draw_circles (nvosdpadding->nvosdpadding_context,
            nvosdpadding->frame_circle_params) == -1) {
      // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
      //     ("Unable to draw circles"), NULL);
      return GST_FLOW_ERROR;
    }
  }

  if (nvosdpadding->nvosdpadding_mode == MODE_GPU) {
    if (nvll_osd_apply (nvosdpadding->nvosdpadding_context, NULL, surface) == -1) {
      // GST_ELEMENT_ERROR (nvosdpadding, RESOURCE, FAILED,
      //     ("Unable to draw shapes onto video frame by GPU"), NULL);
      return GST_FLOW_ERROR;
    }
  }

  nvtxRangePop ();
  nvosdpadding->frame_num++;

  nvds_set_output_system_timestamp (buf, GST_ELEMENT_NAME (nvosdpadding));

  gst_buffer_unmap (buf, &inmap);
  return GST_FLOW_OK;
}

/* Called when the plugin is destroyed.
 * Free all structures which have been malloc'd.
 */
static void
GST_NVOSDPADDING_finalize (GObject * object)
{
  GstNvOsdPadding *nvosdpadding = GST_NVOSDPADDING (object);

  if (nvosdpadding->clock_text_params.font_params.font_name) {
    g_free ((char *) nvosdpadding->clock_text_params.font_params.font_name);
  }
  g_free (nvosdpadding->rect_params);
  g_free (nvosdpadding->mask_rect_params);
  g_free (nvosdpadding->mask_params);
  g_free (nvosdpadding->text_params);
  g_free (nvosdpadding->line_params);
  g_free (nvosdpadding->arrow_params);
  g_free (nvosdpadding->circle_params);

  g_free (nvosdpadding->frame_rect_params);
  g_free (nvosdpadding->frame_mask_params);
  g_free (nvosdpadding->frame_text_params);
  g_free (nvosdpadding->frame_line_params);
  g_free (nvosdpadding->frame_arrow_params);
  g_free (nvosdpadding->frame_circle_params);

  G_OBJECT_CLASS (parent_class)->finalize (object);
}

/* Install properties, set sink and src pad capabilities, override the required
 * functions of the base class, These are common to all instances of the
 * element.
 */
static void
GST_NVOSDPADDING_class_init (GstNvOsdPaddingClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseTransformClass *base_transform_class =
      GST_BASE_TRANSFORM_CLASS (klass);

  gobject_class = (GObjectClass *) klass;
  gstelement_class = (GstElementClass *) klass;

  base_transform_class->transform_ip =
      GST_DEBUG_FUNCPTR (GST_NVOSDPADDING_transform_ip);
  base_transform_class->start = GST_DEBUG_FUNCPTR (GST_NVOSDPADDING_start);
  base_transform_class->stop = GST_DEBUG_FUNCPTR (GST_NVOSDPADDING_stop);
  base_transform_class->set_caps = GST_DEBUG_FUNCPTR (GST_NVOSDPADDING_set_caps);
  base_transform_class->transform_caps =
      GST_DEBUG_FUNCPTR (GST_NVOSDPADDING_transform_caps);
  base_transform_class->sink_event = GST_DEBUG_FUNCPTR (GST_NVOSDPADDING_sink_event);

  gobject_class->set_property = GST_NVOSDPADDING_set_property;
  gobject_class->get_property = GST_NVOSDPADDING_get_property;
  gobject_class->finalize = GST_NVOSDPADDING_finalize;

  base_transform_class->passthrough_on_same_caps = TRUE;

  g_object_class_install_property (gobject_class, PROP_PADDING_SIZE,
      g_param_spec_uint ("padding-size", "Padding Size",
          "Size of padding to add around the frame", 
          0, G_MAXUINT, DEFAULT_PADDING_SIZE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_ENABLE_PADDING,
      g_param_spec_boolean ("enable-padding", "Enable Padding",
          "Enable padding with background color",
          DEFAULT_ENABLE_PADDING,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_PADDING_COLOR,
      g_param_spec_string ("padding-color", "Padding Color",
          "RGBA padding color (format: R,G,B,A)",
          "255,255,255,255",
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_SHOW_CLOCK,
      g_param_spec_boolean ("display-clock", "clock",
          "Whether to display clock", FALSE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_SHOW_TEXT,
      g_param_spec_boolean ("display-text", "text",
          "Whether to display text", TRUE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_SHOW_BBOX,
      g_param_spec_boolean ("display-bbox", "text",
          "Whether to display bounding boxes", TRUE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_SHOW_MASK,
      g_param_spec_boolean ("display-mask", "text",
          "Whether to display instance mask", TRUE, G_PARAM_READWRITE));

  g_object_class_install_property (gobject_class, PROP_CLOCK_FONT,
      g_param_spec_string ("clock-font", "clock-font",
          "Clock Font to be set",
          "DEFAULT_FONT",
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_CLOCK_FONT_SIZE,
      g_param_spec_uint ("clock-font-size", "clock-font-size",
          "font size of the clock",
          0, MAX_FONT_SIZE, DEFAULT_FONT_SIZE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CLOCK_X_OFFSET,
      g_param_spec_uint ("x-clock-offset", "x-clock-offset",
          "x-clock-offset",
          0, G_MAXUINT, 0,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CLOCK_Y_OFFSET,
      g_param_spec_uint ("y-clock-offset", "y-clock-offset",
          "y-clock-offset",
          0, G_MAXUINT, 0,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_CLOCK_COLOR,
      g_param_spec_uint ("clock-color", "clock-color",
          "clock-color",
          0, G_MAXUINT, G_MAXUINT,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_PROCESS_MODE,
      g_param_spec_enum ("process-mode", "Process Mode",
          "Rect and text draw process mode, CPU_MODE only support RGBA format",
          GST_TYPE_NV_OSD_PROCESS_MODE,
          GST_NV_OSD_DEFAULT_PROCESS_MODE,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));

  g_object_class_install_property (gobject_class, PROP_GPU_DEVICE_ID,
      g_param_spec_uint ("gpu-id", "Set GPU Device ID",
          "Set GPU Device ID",
          0, G_MAXUINT, 0,
          (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS |
          GST_PARAM_MUTABLE_READY)));

  gst_element_class_set_details_simple (gstelement_class,
      "nvosdpadding plugin",
      "nvosdpadding functionality",
      "Gstreamer bounding box draw element",
      "NVIDIA Corporation. Post on Deepstream for Tesla forum for any queries "
      "@ https://devtalk.nvidia.com/default/board/209/");

  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&sink_factory));
  gst_element_class_add_pad_template (gstelement_class,
      gst_static_pad_template_get (&src_factory));

  _dsmeta_quark = g_quark_from_static_string (NVDS_META_STRING);
}

/* Function called when a property of the element is set. Standard boilerplate.
 */
static void
GST_NVOSDPADDING_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstNvOsdPadding *nvosdpadding = GST_NVOSDPADDING (object);

  switch (prop_id) {
    case PROP_ENABLE_PADDING:
      nvosdpadding->enable_padding = g_value_get_boolean (value);
      break;
    case PROP_PADDING_SIZE:
      nvosdpadding->padding_size = g_value_get_uint(value);
      break;

    case PROP_PADDING_COLOR:
    {
      const gchar *color_str = g_value_get_string (value);
      gchar **color_values = g_strsplit(color_str, ",", 4);
      if (color_values[0] && color_values[1] && color_values[2] && color_values[3]) {
        nvosdpadding->padding_color[0] = atoi(color_values[0]); // R
        nvosdpadding->padding_color[1] = atoi(color_values[1]); // G
        nvosdpadding->padding_color[2] = atoi(color_values[2]); // B
        nvosdpadding->padding_color[3] = atoi(color_values[3]); // A
      }
      g_strfreev(color_values);
      break;
    }
    case PROP_SHOW_CLOCK:
      nvosdpadding->show_clock = g_value_get_boolean (value);
      break;
    case PROP_SHOW_TEXT:
      nvosdpadding->draw_text = g_value_get_boolean (value);
      break;
    case PROP_SHOW_BBOX:
      nvosdpadding->draw_bbox = g_value_get_boolean (value);
      break;
    case PROP_SHOW_MASK:
      nvosdpadding->draw_mask = g_value_get_boolean (value);
      break;
    case PROP_CLOCK_FONT:
      if (nvosdpadding->clock_text_params.font_params.font_name) {
        g_free ((char *) nvosdpadding->clock_text_params.font_params.font_name);
      }
      nvosdpadding->clock_text_params.font_params.font_name =
          (gchar *) g_value_dup_string (value);
      break;
    case PROP_CLOCK_FONT_SIZE:
      nvosdpadding->clock_text_params.font_params.font_size =
          g_value_get_uint (value);
      break;
    case PROP_CLOCK_X_OFFSET:
      nvosdpadding->clock_text_params.x_offset = g_value_get_uint (value);
      break;
    case PROP_CLOCK_Y_OFFSET:
      nvosdpadding->clock_text_params.y_offset = g_value_get_uint (value);
      break;
    case PROP_CLOCK_COLOR:
      GST_NVOSDPADDING_parse_color (nvosdpadding, g_value_get_uint (value));
      break;
    case PROP_PROCESS_MODE:
      nvosdpadding->nvosdpadding_mode = (NvOSD_Mode) g_value_get_enum (value);
      if (nvosdpadding->nvosdpadding_mode > MODE_GPU) {
        g_print ("WARN !! Invalid mode selected, Falling back to GPU\n");
        nvosdpadding->nvosdpadding_mode =
            nvosdpadding->nvosdpadding_mode > MODE_GPU ? MODE_GPU : nvosdpadding->nvosdpadding_mode;
      }
      break;
    case PROP_GPU_DEVICE_ID:
      nvosdpadding->gpu_id = g_value_get_uint (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Function called when a property of the element is requested. Standard
 * boilerplate.
 */
static void
GST_NVOSDPADDING_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstNvOsdPadding *nvosdpadding = GST_NVOSDPADDING (object);

  switch (prop_id) {
    case PROP_ENABLE_PADDING:
      g_value_set_boolean (value, nvosdpadding->enable_padding);
      break;
    case PROP_PADDING_COLOR:
    {
      gchar *color_str = g_strdup_printf("%d,%d,%d,%d",
          nvosdpadding->padding_color[0],
          nvosdpadding->padding_color[1],
          nvosdpadding->padding_color[2],
          nvosdpadding->padding_color[3]);
      g_value_set_string (value, color_str);
      g_free(color_str);
      break;
    }
    case PROP_SHOW_CLOCK:
      g_value_set_boolean (value, nvosdpadding->show_clock);
      break;
    case PROP_SHOW_TEXT:
      g_value_set_boolean (value, nvosdpadding->draw_text);
      break;
    case PROP_SHOW_BBOX:
      g_value_set_boolean (value, nvosdpadding->draw_bbox);
      break;
    case PROP_SHOW_MASK:
      g_value_set_boolean (value, nvosdpadding->draw_mask);
      break;
    case PROP_CLOCK_FONT:
      g_value_set_string (value, nvosdpadding->font);
      break;
    case PROP_CLOCK_FONT_SIZE:
      g_value_set_uint (value, nvosdpadding->clock_font_size);
      break;
    case PROP_CLOCK_X_OFFSET:
      g_value_set_uint (value, nvosdpadding->clock_text_params.x_offset);
      break;
    case PROP_CLOCK_Y_OFFSET:
      g_value_set_uint (value, nvosdpadding->clock_text_params.y_offset);
      break;
    case PROP_CLOCK_COLOR:
      g_value_set_uint (value, nvosdpadding->clock_color);
      break;
    case PROP_PROCESS_MODE:
      g_value_set_enum (value, nvosdpadding->nvosdpadding_mode);
      break;
    case PROP_GPU_DEVICE_ID:
      g_value_set_uint (value, nvosdpadding->gpu_id);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
}

/* Set default values of certain properties.
 */
static void
GST_NVOSDPADDING_init (GstNvOsdPadding * nvosdpadding)
{
  nvosdpadding->show_clock = FALSE;
  nvosdpadding->draw_text = TRUE;
  nvosdpadding->draw_bbox = TRUE;
  nvosdpadding->draw_mask = FALSE;
  nvosdpadding->clock_text_params.font_params.font_name = g_strdup (DEFAULT_FONT);
  nvosdpadding->clock_text_params.font_params.font_size = DEFAULT_FONT_SIZE;
  nvosdpadding->nvosdpadding_mode = GST_NV_OSD_DEFAULT_PROCESS_MODE;
  nvosdpadding->border_width = DEFAULT_BORDER_WIDTH;
  nvosdpadding->num_rect = 0;
  nvosdpadding->num_segments = 0;
  nvosdpadding->num_strings = 0;
  nvosdpadding->num_lines = 0;
  nvosdpadding->clock_text_params.font_params.font_color.red = 1.0;
  nvosdpadding->clock_text_params.font_params.font_color.green = 0.0;
  nvosdpadding->clock_text_params.font_params.font_color.blue = 0.0;
  nvosdpadding->clock_text_params.font_params.font_color.alpha = 1.0;
  nvosdpadding->rect_params = g_new0 (NvOSD_RectParams, MAX_OSD_ELEMS);
  nvosdpadding->mask_rect_params = g_new0 (NvOSD_RectParams, MAX_OSD_ELEMS);
  nvosdpadding->mask_params = g_new0 (NvOSD_MaskParams, MAX_OSD_ELEMS);
  nvosdpadding->text_params = g_new0 (NvOSD_TextParams, MAX_OSD_ELEMS);
  nvosdpadding->line_params = g_new0 (NvOSD_LineParams, MAX_OSD_ELEMS);
  nvosdpadding->arrow_params = g_new0 (NvOSD_ArrowParams, MAX_OSD_ELEMS);
  nvosdpadding->circle_params = g_new0 (NvOSD_CircleParams, MAX_OSD_ELEMS);
  nvosdpadding->frame_rect_params = g_new0 (NvOSD_FrameRectParams, MAX_OSD_ELEMS);
  nvosdpadding->frame_mask_params =
      g_new0 (NvOSD_FrameSegmentMaskParams, MAX_OSD_ELEMS);
  nvosdpadding->frame_text_params = g_new0 (NvOSD_FrameTextParams, MAX_OSD_ELEMS);
  nvosdpadding->frame_line_params = g_new0 (NvOSD_FrameLineParams, MAX_OSD_ELEMS);
  nvosdpadding->frame_arrow_params = g_new0 (NvOSD_FrameArrowParams, MAX_OSD_ELEMS);
  nvosdpadding->frame_circle_params =
      g_new0 (NvOSD_FrameCircleParams, MAX_OSD_ELEMS);
  nvosdpadding->enable_padding = DEFAULT_ENABLE_PADDING;
  nvosdpadding->padding_size = DEFAULT_PADDING_SIZE;
  nvosdpadding->padding_color[0] = DEFAULT_PADDING_COLOR_R;
  nvosdpadding->padding_color[1] = DEFAULT_PADDING_COLOR_G;
  nvosdpadding->padding_color[2] = DEFAULT_PADDING_COLOR_B;
  nvosdpadding->padding_color[3] = DEFAULT_PADDING_COLOR_A;
      
}

/**
 * Set color of text for clock, if enabled.
 */
static gboolean
GST_NVOSDPADDING_parse_color (GstNvOsdPadding * nvosdpadding,
    guint clock_color)
{
  // Update references inside function
  nvosdpadding->clock_text_params.font_params.font_color.red =
      (gfloat) ((clock_color & 0xff000000) >> 24) / 255;
  nvosdpadding->clock_text_params.font_params.font_color.green =
      (gfloat) ((clock_color & 0x00ff0000) >> 16) / 255;
  nvosdpadding->clock_text_params.font_params.font_color.blue =
      (gfloat) ((clock_color & 0x0000ff00) >> 8) / 255;
  nvosdpadding->clock_text_params.font_params.font_color.alpha =
      (gfloat) ((clock_color & 0x000000ff)) / 255;
  return TRUE;
}
/**
 * Boiler plate for registering a plugin and an element.
 */
static gboolean
nvosdpadding_init(GstPlugin * plugin)
{
  GST_DEBUG_CATEGORY_INIT(GST_NVOSDPADDING_debug, "nvosdpadding", 0, "nvosdpadding plugin");

  return gst_element_register(plugin, "nvosdpadding", GST_RANK_PRIMARY,
      GST_TYPE_NVOSDPADDING);
}

#ifndef PACKAGE
#define PACKAGE "nvosdpadding"
#endif

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    nvosdpadding,
    PACKAGE_DESCRIPTION,
    nvosdpadding_init,
    "1.0",
    PACKAGE_LICENSE,
    PACKAGE_NAME,
    PACKAGE_URL)
