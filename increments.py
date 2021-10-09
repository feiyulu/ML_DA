import numpy as np
import xarray as xr
import os, os.path
import glob
from cartopy import crs as ccrs
from cartopy import feature
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from io_util import set_output_dir
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import calendar

work_dir="/work/Feiyu.Lu/"
inc_dir=work_dir+"increments/"
ecda_dir="/archive/Feiyu.Lu/SPEAR/SPEAR_c96_o1_ECDA_"
    
static_file='/work/Feiyu.Lu/SPEAR/ocean_z.static.nc'
static_ds=xr.open_dataset(static_file)
area_t=static_ds.area_t.where(static_ds.wet)
xh,yh=np.meshgrid(static_ds.xh, static_ds.yh)

thickness_file="/work/Feiyu.Lu/ECDA_data/vgrid_75_2m.nc"
thickness_ds=xr.open_dataset(thickness_file)
thickness=thickness_ds.dz
# print(thickness)
# print(thickness[0:30].sum())
sec_per_julian_year=3600*24*365.25
sec_per_day=3600*24

static_ds=xr.open_dataset("/work/Feiyu.Lu/SPEAR/ocean_z.static.nc")
area=static_ds.areacello
geolat=static_ds.geolat
geolon=static_ds.geolon

basin=xr.open_dataarray('/work/Feiyu.Lu/ECDA_data/basin.nc')
# print(basin)
basin_code=dict(zip(['SO','Atlantic','Pacific','Arctic','Indian','Mediterranean'],[1,2,3,4,5,6]))

regions={
    "Global":dict(lats=slice(-80,80),lons=slice(-300,60),proj=ccrs.Robinson(210),PanelHeight=2.2),
    "Global_60":dict(lats=slice(-60,60),lons=slice(-300,60),proj=ccrs.Robinson(210),PanelHeight=2),
    "TropPac":dict(lats=slice(-30,30),lons=slice(-250,-70),proj=ccrs.Robinson(200),PanelHeight=2),
    "TropAtl":dict(lats=slice(-30,30),lons=slice(-75,25),proj=ccrs.Robinson(-25),PanelHeight=2.5),
    "NorthAtl":dict(lats=slice(20,70),lons=slice(-80,0),proj=ccrs.Robinson(-45),PanelHeight=3),
    "Atlantic":dict(lats=slice(-40,60),lons=slice(-85,15),proj=ccrs.Robinson(-35),PanelHeight=4),
    "Arctic":dict(lats=slice(55,90),lons=slice(-300,60),proj=ccrs.NorthPolarStereo(0),PanelHeight=4),
    "SO":dict(lats=slice(-90,-50),lons=slice(-300,60),proj=ccrs.SouthPolarStereo(0),PanelHeight=4),
    'Kuroshio':dict(lats=slice(20,60),lons=slice(-260,-180),proj=ccrs.Robinson(140),PanelHeight=2),
    "Nino4":dict(lats=slice(-5,5),lons=slice(-200,-150)),
    "Nino34":dict(lats=slice(-5,5),lons=slice(-170,-120)),
    "Nino3":dict(lats=slice(-5,5),lons=slice(-150,-90)),
    "ATL3":dict(lats=slice(-3,3),lons=slice(-20,0)),
    "Nino12":dict(lats=slice(-10,0),lons=slice(-90,-80))
}

def rightmost_colorbar_subplots(figsize,n_row,n_col,proj):
    fig=plt.figure(figsize=figsize)
    aspect_ratio=(figsize[1]/n_row)/(figsize[0]/n_col)
    last_col=1+aspect_ratio/2
    fig_gs=gs.GridSpec(nrows=n_row, ncols=n_col, width_ratios=[1 for i in range(n_col-1)]+[last_col])
    axes=[]
    for i in range(n_row):
        for j in range(n_col):
            if proj:
                axes.append(plt.subplot(fig_gs[i,j],projection=proj))
            else:
                axes.append(plt.subplot(fig_gs[i,j]))
    
    return fig,axes

def pcolormesh_cm_scale(clim,cm='bwr',scale=2,n=6,offset=1):
    levels=[-clim/(offset*scale**i) for i in range(0,n)]+[clim/(offset*scale**(n-1-i)) for i in range(0,n)]
    cm = plt.get_cmap(cm)
    N=len(levels)
    norm = BoundaryNorm(levels, ncolors=cm.N)
    return norm
    
def annual_mean_latlon_plot(var,region,exp_name,depths,offset=1,clim=None):
    inc_file="inc."+exp_name+".2007-2018.mean.nc"
    inc_ds=xr.open_dataset("/work/Feiyu.Lu/increments/"+exp_name+"/"+inc_file)  
    inc_var=inc_ds[var+'_increment']*sec_per_day
    
    n_plots=len(depths)
    n_row=np.ceil(np.sqrt(n_plots))
    n_col=np.ceil(n_plots/n_row)
    fig,axes=plt.subplots(int(n_row),int(n_col),figsize=[15,3/n_col*regions[region]['PanelHeight']*n_row],
                          subplot_kw=dict(projection=regions[region]['proj']))
    for depth,ax in zip(depths,axes.flat):
        inc_slice=inc_var.sel(yh=regions[region]['lats'],xh=regions[region]['lons']).sel(z_l=depth,method='nearest')
        if clim is None:
            clim=np.abs(inc_slice).max().values
        norm = pcolormesh_cm_scale(clim,'bwr',offset=offset,scale=2)
        cs=ax.pcolormesh(geolon.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                         geolat.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                         inc_slice.squeeze(),norm=norm,cmap='bwr',transform=ccrs.PlateCarree())
        if clim>0.1:
            fig.colorbar(cs, ax=ax, format='%3.2f')
        else:
            fig.colorbar(cs, ax=ax, format='%.2e')
        cs.colorbar.set_label(inc_ds[var+'_increment'].attrs['units']+'/day')
        ax.coastlines()
        ax.set_title('{0} increments at {1:.1f}m'.format(var,inc_slice['z_l'].values))
    
    output_dir=set_output_dir(['increments',exp_name])
    plt.savefig(output_dir+"/annual_mean_"+var+'_'+region+'_'+str(depths[0])+'-'+str(depths[-1])+'.jpg')

    return fig,axes

def annual_cycle_latlon_plot(var,region,exp_name,depth,months=range(1,13),offset=1,clim=None):
    inc_file="inc."+exp_name+".2007-2018.raw.nc"
    inc_ds=xr.open_dataset("/work/Feiyu.Lu/increments/"+exp_name+"/"+inc_file)  
    inc_var=inc_ds[var+'_increment']*sec_per_day
    inc_slice=inc_var.sel(yh=regions[region]['lats'],xh=regions[region]['lons']).sel(z_l=depth,method='nearest')
    if clim is None:
        clim=np.abs(inc_slice).max().values
    norm = pcolormesh_cm_scale(clim,offset=offset)
    
    n_plots=len(months)
    n_row=np.ceil(np.sqrt(n_plots))
    n_col=np.ceil(n_plots/n_row)
    fig,axes=rightmost_colorbar_subplots([15,3/n_col*regions[region]['PanelHeight']*n_row],
                                        int(n_row),int(n_col),regions[region]['proj'])
    for i,month in enumerate(months):
        cs=axes[i].pcolormesh(geolon.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                         geolat.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                         inc_slice.isel(time=month-1).squeeze(),norm=norm,cmap='bwr',
                              transform=ccrs.PlateCarree())
        if (i+1)%n_col==0:
            if clim>0.1:
                fig.colorbar(cs, ax=axes[i], format='%3.2f')
            else:
                fig.colorbar(cs, ax=axes[i], format='%.2e')
            cs.colorbar.set_label(inc_ds[var+'_increment'].attrs['units']+'/year')

        axes[i].coastlines()
        axes[i].set_title('{3} {2} {0}_inc at {1:.1f}m'.format(var,inc_slice['z_l'].values,
                                                           calendar.month_abbr[month],exp_name))
        
    output_dir=set_output_dir(['increments',exp_name])
    plt.savefig(output_dir+"/annual_cycle_"+var+'_'+region+'_'+str(depth)+'.jpg')
        
    return fig,axes

def annual_mean_latlon_diff_plot(var,region,exp_name1,exp_name2,depths,offset=1):
    inc_file1="inc."+exp_name1+".2007-2018.mean.nc"
    inc_ds1=xr.open_dataset("/work/Feiyu.Lu/increments/"+exp_name1+"/"+inc_file1)  
    inc_var1=inc_ds1[var+'_increment']*sec_per_julian_year
    
    inc_file2="inc."+exp_name2+".2007-2018.mean.nc"
    inc_ds2=xr.open_dataset("/work/Feiyu.Lu/increments/"+exp_name2+"/"+inc_file2)  
    inc_var2=inc_ds2[var+'_increment']*sec_per_julian_year
    
    n_col,n_row=3,len(depths)
    fig,axes=rightmost_colorbar_subplots([15,regions[region]['PanelHeight']*n_row],
                                        int(n_row),int(n_col),regions[region]['proj'])
    for i,depth in enumerate(depths):
        inc_slice1=inc_var1.sel(yh=regions[region]['lats'],xh=regions[region]['lons']).sel(z_l=depth,method='nearest')
        inc_slice2=inc_var2.sel(yh=regions[region]['lats'],xh=regions[region]['lons']).sel(z_l=depth,method='nearest')
        
        clim=np.max([np.abs(inc_slice1).max().values,np.abs(inc_slice2).max().values])
        norm = pcolormesh_cm_scale(clim,'bwr',offset=offset)
        
        axes[i*3].pcolormesh(geolon.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                             geolat.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                             inc_slice1.squeeze(),norm=norm,cmap='bwr',transform=ccrs.PlateCarree())
        axes[i*3+1].pcolormesh(geolon.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                             geolat.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                             inc_slice2.squeeze(),norm=norm,cmap='bwr',transform=ccrs.PlateCarree())
        cs=axes[i*3+2].pcolormesh(geolon.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                                geolat.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                                (inc_slice1-inc_slice2).squeeze(),norm=norm,cmap='bwr',transform=ccrs.PlateCarree())
        if clim>0.5:
            fig.colorbar(cs, ax=axes[i*3+2], format='%1.2f')
        else:
            fig.colorbar(cs, ax=axes[i*3+2], format='%.1e')
            
        cs.colorbar.set_label(inc_ds1[var+'_increment'].attrs['units']+'/year')
        
        axes[i*3].set_title('{0} {1} increments at {2:.1f}m'.format(exp_name1,var,inc_slice1['z_l'].values))
        axes[i*3+1].set_title('{0} {1} increments at {2:.1f}m'.format(exp_name2,var,inc_slice2['z_l'].values))
        axes[i*3+2].set_title('{0} - {1}'.format(exp_name1,exp_name2))
        
    for ax in axes:
        ax.coastlines()     
        
    plt.savefig('/home/Feiyu.Lu/Documents/SPEAR_ECDA/increments/'+exp_name1+
                "/annual_mean_diff_"+exp_name2+'_'+var+'_'+region+'_'+str(depths[0])+'-'+str(depths[-1])+'.jpg')

    return fig,axes

def annual_cycle_latlon_diff_plot(var,region,exp_name1,exp_name2,depth,months=[1,4,7,10],offset=1):
    inc_file1="inc."+exp_name1+".2007-2018.raw.nc"
    inc_ds1=xr.open_dataset("/work/Feiyu.Lu/increments/"+exp_name1+"/"+inc_file1)  
    inc_var1=inc_ds1[var+'_increment']*sec_per_julian_year
    inc_slice1=inc_var1.sel(yh=regions[region]['lats'],xh=regions[region]['lons']).sel(z_l=depth,method='nearest')

    inc_file2="inc."+exp_name2+".2007-2018.raw.nc"
    inc_ds2=xr.open_dataset("/work/Feiyu.Lu/increments/"+exp_name2+"/"+inc_file2)  
    inc_var2=inc_ds2[var+'_increment']*sec_per_julian_year
    inc_slice2=inc_var2.sel(yh=regions[region]['lats'],xh=regions[region]['lons']).sel(z_l=depth,method='nearest')

    clim=np.max([np.abs(inc_slice1).max().values,np.abs(inc_slice2).max().values])
    norm = pcolormesh_cm_scale(clim,'bwr',offset=offset)
    
    n_col,n_row=3,len(months)
    fig,axes=rightmost_colorbar_subplots([15,regions[region]['PanelHeight']*n_row],
                                        int(n_row),int(n_col),regions[region]['proj'])
    for i,month in enumerate(months):
        axes[i*3].pcolormesh(geolon.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                             geolat.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                             inc_slice1.isel(time=month-1).squeeze(),
                             norm=norm,cmap='bwr',transform=ccrs.PlateCarree())
        axes[i*3+1].pcolormesh(geolon.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                             geolat.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                             inc_slice2.isel(time=month-1).squeeze(),
                             norm=norm,cmap='bwr',transform=ccrs.PlateCarree())
        cs=axes[i*3+2].pcolormesh(geolon.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                                geolat.sel(yh=regions[region]['lats'],xh=regions[region]['lons']),
                                (inc_slice1-inc_slice2).isel(time=month-1).squeeze(),
                                norm=norm,cmap='bwr',transform=ccrs.PlateCarree())           
        if clim>0.5:
            fig.colorbar(cs, ax=axes[i*3+2], format='%2.1f')
        else:
            fig.colorbar(cs, ax=axes[i*3+2], format='%.1e')  
        cs.colorbar.set_label(inc_ds1[var+'_increment'].attrs['units']+'/year')
        
        for ax in axes:
            ax.coastlines()
            
        axes[i*3].set_title('{3} {2} {0}_inc at {1:.1f}m'.format(var,inc_slice1['z_l'].values,
                                                                 calendar.month_abbr[month],exp_name1))
        axes[i*3+1].set_title('{3} {2} {0}_inc at {1:.1f}m'.format(var,inc_slice2['z_l'].values,
                                                                   calendar.month_abbr[month],exp_name2))
        axes[i*3+2].set_title('{0} - {1}'.format(exp_name1,exp_name2))
        
    plt.savefig('/home/Feiyu.Lu/Documents/SPEAR_ECDA/increments/'+exp_name1+
                "/annual_cycle_diff_"+exp_name2+'_'+var+'_'+region+'_'+str(depth)+'.jpg')
        
    return fig,axes

def annual_mean_depthlon_plot(var,region,exp_name,depths,offset=1,figsize=[10,6]):
    inc_file="inc."+exp_name+".2007-2018.mean.nc"
    inc_ds=xr.open_dataset("/work/Feiyu.Lu/increments/"+exp_name+"/"+inc_file)  
    inc_var=inc_ds[var+'_increment']*sec_per_julian_year
    
    fig,ax=plt.subplots(1,1,figsize=figsize)
    
    inc_slice=inc_var.sel(yh=regions[region]['lats'],xh=regions[region]['lons'],z_l=depths).mean('yh')
    clim=np.abs(inc_slice).max().values
    norm = pcolormesh_cm_scale(clim,'bwr',offset=offset)
    cs=ax.pcolormesh(inc_slice.xh,inc_slice.z_l,inc_slice.squeeze(),norm=norm,cmap='bwr')
    if clim>0.5:
        fig.colorbar(cs, ax=ax, format='%2.1f')
    else:
        fig.colorbar(cs, ax=ax, format='%.1e')
    cs.colorbar.set_label(inc_ds[var+'_increment'].attrs['units']+'/year')
    ax.invert_yaxis()
    ax.set_title('{1} {0} increments'.format(var,exp_name))
    ax.set_ylabel('depth(m)')
    ax.set_xlabel('lon')
    
    output_dir=set_output_dir(['increments',exp_name])
    plt.savefig(output_dir+"/annual_mean_xz_"+var+'_'+region+'_'+str(depths.stop)+'.jpg')

    return fig,ax

def annual_cycle_depthlon_plot(var,region,exp_name,depths,months=range(1,13),offset=1):
    inc_file="inc."+exp_name+".2007-2018.raw.nc"
    inc_ds=xr.open_dataset("/work/Feiyu.Lu/increments/"+exp_name+"/"+inc_file)  
    inc_var=inc_ds[var+'_increment']*sec_per_julian_year
    area_slice=area.sel(yh=regions[region]['lats'],xh=regions[region]['lons'])
    area_wt=area_slice/area_slice.sum('yh')
    inc_slice=(inc_var.sel(yh=regions[region]['lats'],xh=regions[region]['lons'],z_l=depths)*area_wt).sum('yh')
    clim=np.abs(inc_slice).max().values
    norm = pcolormesh_cm_scale(clim,offset=offset)
    
    n_plots=len(months)
    n_row=np.ceil(np.sqrt(n_plots))
    n_col=np.ceil(n_plots/n_row)
    fig,axes=rightmost_colorbar_subplots([15,9/n_col*n_row],int(n_row),int(n_col))
    for i,month in enumerate(months):
        cs=axes[i].pcolormesh(inc_slice.xh,inc_slice.z_l,inc_slice.isel(time=month-1).squeeze(),
                              norm=norm,cmap='bwr')
        if (i+1)%n_col==0:
            if clim>0.5:
                fig.colorbar(cs, ax=axes[i], format='%2.1f')
            else:
                fig.colorbar(cs, ax=axes[i], format='%.1e')
            cs.colorbar.set_label(inc_ds[var+'_increment'].attrs['units']+'/year')
        axes[i].invert_yaxis()
        axes[i].set_ylabel('depth(m)')
        axes[i].set_xlabel('lon')
        axes[i].set_title('{2} {1} {0}_inc'.format(var,calendar.month_abbr[month],exp_name))
        
    for ax in axes[0:-int(n_col)]:
        ax.set_xticks([])
        ax.set_xlabel('')
        
    output_dir=set_output_dir(['increments',exp_name])
    plt.savefig(output_dir+"/annual_cycle_xz_"+var+'_'+region+'_'+str(depths.stop)+'.jpg')
        
    return fig,axes

def annual_cycle_depth_plot(var,region,exp_name,depths,offset=1,figsize=[10,6]):
    inc_file="inc."+exp_name+".2007-2018.raw.nc"
    inc_ds=xr.open_dataset("/work/Feiyu.Lu/increments/"+exp_name+"/"+inc_file)  
    inc_var=inc_ds[var+'_increment']*sec_per_julian_year
    area_slice=area.sel(yh=regions[region]['lats'],xh=regions[region]['lons'])
    area_wt=area_slice/area_slice.sum(('xh','yh'))
    inc_ave=(inc_var.sel(yh=regions[region]['lats'],xh=regions[region]['lons'],z_l=depths)*area_wt).sum(('xh','yh'))
    clim=np.abs(inc_ave).max().values
    norm = pcolormesh_cm_scale(clim,offset=offset)
    
    fig,ax=plt.subplots(1,1,figsize=figsize)
    cs=ax.pcolormesh(np.arange(0.5,13,1),inc_ave.z_l,inc_ave.T,norm=norm,cmap='bwr')
    if clim>0.5:
        fig.colorbar(cs, ax=ax, format='%2.1f')
    else:
        fig.colorbar(cs, ax=ax, format='%.1e')
    cs.colorbar.set_label(inc_ds[var+'_increment'].attrs['units']+'/year')
    ax.invert_yaxis()
    ax.set_ylabel('depth(m)')
    ax.set_xlabel('Month')
    ax.set_xticks(range(1,13))
    ax.set_xticklabels([calendar.month_abbr[month] for month in range(1,13)])
    ax.set_title('{2} {1} {0}_inc Annual Cycle'.format(var,region,exp_name))
               
    output_dir=set_output_dir(['increments',exp_name])
    plt.savefig(output_dir+"/annual_cycle_z_"+var+'_'+region+'_'+str(depths.stop)+'.jpg')
        
    return fig,ax

def ts_depth_plot(var,region,exp_name,depths,
                  time_slice=slice('20070101','20181231'),offset=1,figsize=[15,6]):
    inc_file="inc.month.nc"
    inc_ds=xr.open_dataset("/work/Feiyu.Lu/increments/"+exp_name+"/"+inc_file)  
    inc_var=inc_ds[var+'_increment']*sec_per_julian_year
    inc_slice=inc_var.sel(time=time_slice,yh=regions[region]['lats'],xh=regions[region]['lons'],z_l=depths)
    area_slice=area.sel(yh=regions[region]['lats'],xh=regions[region]['lons'])
    area_wt=area_slice/area_slice.sum(('xh','yh'))
    inc_ave=(inc_slice*area_wt).sum(('xh','yh'))
    clim=np.abs(inc_ave).max().values
    norm = pcolormesh_cm_scale(clim,offset=offset)
    
    fig,ax=plt.subplots(1,1,figsize=figsize)
    cs=ax.pcolormesh(inc_ave.time,inc_ave.z_l,inc_ave.T,norm=norm,cmap='bwr')
    if clim>0.5:
        fig.colorbar(cs, ax=ax, format='%2.1f')
    else:
        fig.colorbar(cs, ax=ax, format='%.1e')
    cs.colorbar.set_label(inc_ds[var+'_increment'].attrs['units']+'/year')
    ax.invert_yaxis()
    ax.set_ylabel('depth(m)')
    ax.set_xlabel('lon')
    ax.set_title('{2} {1} {0}_inc'.format(var,region,exp_name))
        
    output_dir=set_output_dir(['increments',exp_name])
    plt.savefig(output_dir+"/ts_z_"+var+'_'+region+'_'+str(depths.stop)+'.jpg')
        
    return fig,ax