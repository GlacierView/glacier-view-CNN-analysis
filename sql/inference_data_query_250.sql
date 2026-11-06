-- Every qualifying summer image for the 250-per-region glacier selection.
--
-- Output is one row per file in s3://full-time-series-images-t1-l2-sr/
-- (us-west-1), keyed {glims_id}/{file_name}. Pass it to get_rasters as
-- filtered_file_path, or use it to scope an `aws s3 cp`.
--
-- Two steps in one query:
--
--   selected_glaciers  identify_inference_glims_ids_geog_area_rollup_250.sql,
--                      verbatim apart from projecting glims_id and its rank.
--                      Reads the *training* tables. 1,024 glaciers.
--   the body           inference_data_query.sql pointed at the
--                      _full_time_series_250 tables, with rank_score still
--                      commented out so the whole time series comes back.
--
-- Glaciers are restricted by joining on selected_glaciers, NOT by re-ranking
-- with geog_size_rank <= 250 inside the _250 tables. The _250 tables hold
-- 1,101 glaciers, 77 more than were selected, and re-ranking over that larger
-- population swaps 23 glaciers in Asia and South America. geog_size_rank in
-- the output is the rank from the selection step.
--
-- Verified 2026-10-30: 68,880 files across all 1,024 glaciers (7 to 324 per
-- glacier, median 53), every one present in the bucket, 57.5 GB.
--
-- Run against the `glacier-view` database in us-west-1.

with selected_glaciers as (
    select
        glims_id
        , geog_size_rank
    from (
        select
            *
            , dense_rank() over(partition by geog_area_rollup order by db_area desc) geog_size_rank
        from (
            select
                *
                , row_number() over(partition by glims_id order by src_date desc) as rank_score
                ,   case
                    when
                        is_southen_hemisphere = 0
                        and (month_number >= 5  and month_number <= 10)
                        then 1
                    when
                        is_southen_hemisphere = 1
                        and (month_number >= 11 or month_number <= 4)
                        then 1
                    end as is_low_snow_month
            from (
                select
                    eem.file_name
                    , eem.glims_id
                    , eem.cloud_cover
                    , eem.image_quality
                    , eem.image_quality_oli
                    , eem.spacecraft_id
                    , cast(eem.src_date as timestamp) as src_date
                    , month(cast(eem.src_date as timestamp)) as month_number
                    , ia.height_in_pixels*ia.width_in_pixels*num_of_bands as num_pixels
                    , ia.no_data_pixel_count
                    , ia.zero_pixel_count
                    , cast(ia.zero_pixel_count as decimal(38,19))/(ia.height_in_pixels*ia.width_in_pixels*num_of_bands) as percentage_zero_pixels
                    , ia.negative_pixel_count
                    , glims_18k.geog_area_rollup
                    , glims_18k.geog_area
                    , glims_18k.db_area
                    , case
                        when glims_18k.geog_area_rollup in ('South America', 'Oceania') then 1
                        else 0
                    end as is_southen_hemisphere
                from
                    ee_metadata eem
                    join image_attributes ia on
                        eem.file_name = ia.file_name
                    join glims_18k on
                        eem.glims_id = glims_18k.glac_id
            )
            where
                cloud_cover < 5
                and (image_quality = 9 or image_quality_oli = 9)
                and num_pixels > 50000
                and no_data_pixel_count = 0
                and percentage_zero_pixels < 0.1
                and geog_area != 'Canada' -- this will perfectly exclude glaciers on Ellesmere Island which is too far north for DEMs
        )
        where
            rank_score = 1
            and is_low_snow_month = 1
        )
    where
        geog_size_rank <= 250
)

select
    images.*
    , sg.geog_size_rank
from (
    select
        *
        , row_number() over(partition by glims_id order by src_date desc) as rank_score
        ,   case
            when
                is_southen_hemisphere = 0
                and (month_number >= 5  and month_number <= 10)
                then 1
            when
                is_southen_hemisphere = 1
                and (month_number >= 11 or month_number <= 4)
                then 1
            end as is_low_snow_month
    from (
        select
            eem.file_name
            , eem.glims_id
            , eem.cloud_cover
            , eem.image_quality
            , eem.image_quality_oli
            , eem.spacecraft_id
            , cast(eem.src_date as timestamp) as src_date
            , month(cast(eem.src_date as timestamp)) as month_number
            , ia.height_in_pixels*ia.width_in_pixels*num_of_bands as num_pixels
            , ia.no_data_pixel_count
            , ia.zero_pixel_count
            , cast(ia.zero_pixel_count as decimal(38,19))/(ia.height_in_pixels*ia.width_in_pixels*num_of_bands) as percentage_zero_pixels
            , ia.negative_pixel_count
            , glims_18k.geog_area
            , glims_18k.geog_area_rollup
            , glims_18k.db_area
            , case
                when glims_18k.geog_area_rollup in ('South America', 'Oceania') then 1
                else 0
            end as is_southen_hemisphere
        from
            ee_metadata_full_time_series_250 eem
            join image_attributes_full_time_series_250 ia on
                eem.file_name = ia.file_name
            join glims_18k on
                eem.glims_id = glims_18k.glac_id
    )
    where
        cloud_cover < 5
        and (image_quality = 9 or image_quality_oli = 9)
        and num_pixels > 50000
        and no_data_pixel_count = 0
        and percentage_zero_pixels < 0.1
        and geog_area != 'Canada' -- this will perfectly exclude glaciers on ellesmere island which is too far north for DEMs
) images
join selected_glaciers sg on
    images.glims_id = sg.glims_id
where
   -- rank_score = 1 -- this rank score filter should only be used if we want to have 250 records returned for idenitfying each glacier and a representative time point
    images.is_low_snow_month = 1
