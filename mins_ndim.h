template <class T>
struct F1dim {
	const VecDoub &p;
	const VecDoub &xi;
	Int n;
	T &func;
	VecDoub xt;
	F1dim(VecDoub_I &pp, VecDoub_I &xii, T &funcc) : p(pp),
		xi(xii), n(pp.size()), func(funcc), xt(n) {}
	Doub operator() (const Doub x)
	{
		for (Int j=0;j<n;j++)
			xt[j]=p[j]+x*xi[j];
		return func(xt);
	}
};
template <class T>
struct Linemethod {
	VecDoub p;
	VecDoub xi;
	T &func;
	Int n;
	Linemethod(T &funcc) : func(funcc) {}
	Doub linmin()
	{
		Doub ax,xx,xmin;
		n=p.size();
		F1dim<T> f1dim(p,xi,func);
		ax=0.0;
		xx=1.0;
		Brent brent;
		brent.bracket(ax,xx,f1dim);
		xmin=brent.minimize(f1dim);
		for (Int j=0;j<n;j++) {
			xi[j] *= xmin;
			p[j] += xi[j];
		}
		return brent.fmin;
	}
};
template <class T>
struct Df1dim {
	const VecDoub &p;
	const VecDoub &xi;
	Int n;
	T &funcd;
	VecDoub xt;
	VecDoub dft;
	Df1dim(VecDoub_I &pp, VecDoub_I &xii, T &funcdd) : p(pp),
		xi(xii), n(pp.size()), funcd(funcdd), xt(n), dft(n) {}
	Doub operator()(const Doub x)
	{
		for (Int j=0;j<n;j++)
			xt[j]=p[j]+x*xi[j];
		return funcd(xt);
	}
	Doub df(const Doub x)
	{
		Doub df1=0.0;
		funcd.df(xt,dft);
		for (Int j=0;j<n;j++)
			df1 += dft[j]*xi[j];
		return df1;
	}
};
template <class T>
struct Dlinemethod {
	VecDoub p;
	VecDoub xi;
	T &func;
	Int n;
	Doub LSOTL = 3.0e-8;

	Dlinemethod(T &funcc) : func(funcc) {}
	
	Dlinemethod(T &funcc, Doub LLSOTL) : func(funcc), LSOTL(LLSOTL) {}

	Doub linmin()
	{
		Doub ax,xx,xmin;
		n=p.size();
		Df1dim<T> df1dim(p,xi,func);
		ax=0.0;
		xx=1.0;
		Dbrent dbrent(LSOTL);
		dbrent.bracket(ax,xx,df1dim);
		xmin=dbrent.minimize(df1dim);
		for (Int j=0;j<n;j++) {
			xi[j] *= xmin;
			p[j] += xi[j];
		}
		return dbrent.fmin;
	}
};
template <class T>
struct Powell : Linemethod<T> {
	Int iter;
	Doub fret;
	using Linemethod<T>::func;
	using Linemethod<T>::linmin;
	using Linemethod<T>::p;
	using Linemethod<T>::xi;
	const Doub ftol;
	Powell(T &func, const Doub ftoll=3.0e-8) : Linemethod<T>(func),
		ftol(ftoll) {}
	VecDoub minimize(VecDoub_I &pp)
	{
		Int n=pp.size();
		MatDoub ximat(n,n,0.0);
		for (Int i=0;i<n;i++) ximat[i][i]=1.0;
		return minimize(pp,ximat);
	}
	VecDoub minimize(VecDoub_I &pp, MatDoub_IO &ximat)
	{
		const Int ITMAX=200;
		const Doub TINY=1.0e-25;
		Doub fptt;
		Int n=pp.size();
		p=pp;
		VecDoub pt(n),ptt(n);
		xi.resize(n);
		fret=func(p);
		for (Int j=0;j<n;j++) pt[j]=p[j];
		for (iter=0;;++iter) {
			Doub fp=fret;
			Int ibig=0;
			Doub del=0.0;
			for (Int i=0;i<n;i++) {
				for (Int j=0;j<n;j++) xi[j]=ximat[j][i];
				fptt=fret;
				fret=linmin();
				if (fptt-fret > del) {
					del=fptt-fret;
					ibig=i+1;
				}
			}
			if (2.0*(fp-fret) <= ftol*(abs(fp)+abs(fret))+TINY) {
				return p;
			}
			if (iter == ITMAX) throw("powell exceeding maximum iterations.");
			for (Int j=0;j<n;j++) {
				ptt[j]=2.0*p[j]-pt[j];
				xi[j]=p[j]-pt[j];
				pt[j]=p[j];
			}
			fptt=func(ptt);
			if (fptt < fp) {
				Doub t=2.0*(fp-2.0*fret+fptt)*SQR(fp-fret-del)-del*SQR(fp-fptt);
				if (t < 0.0) {
					fret=linmin();
					for (Int j=0;j<n;j++) {
						ximat[j][ibig-1]=ximat[j][n-1];
						ximat[j][n-1]=xi[j];
					}
				}
			}
		}
	}
};


struct Output_FrprmnV2 {
	int iter;
	Doub final_f;
	Doub final_g2;
	VecDoub final_p;

	Output_FrprmnV2(int iiter, Doub ffinal_f, Doub ffinal_g2, VecDoub & ffinal_p) :
		iter(iiter), final_f(ffinal_f), final_g2(ffinal_g2), final_p(std::move(ffinal_p)) {}
};

template <class T>
struct FrprmnV2 : Dlinemethod<T> {
	Int iter;
	Doub fret;
	using Dlinemethod<T>::func;
	using Dlinemethod<T>::linmin;
	using Dlinemethod<T>::p;
	using Dlinemethod<T>::xi;
	const Doub ftol;
	const Int ITMAX;
	const Doub GTOL; 
	const Doub LSTOL;

	FrprmnV2(T &funcd, const Doub ftoll=3.0e-8, const Doub GGTOL=1.0e-8, const Int IITMAX=200)
	 : Dlinemethod<T>(funcd), ftol(ftoll), GTOL(GGTOL), ITMAX(IITMAX) {
	 	std::cout<<"FTOL: "<< ftol << std::endl;
		std::cout<<"GTOL: "<< GTOL << std::endl;
	}
	
	FrprmnV2(T &funcd, const Doub ftoll, const Doub GGTOL, const Doub LLSTOL, const Int IITMAX)
	 : Dlinemethod<T>(funcd, LLSTOL), ftol(ftoll), GTOL(GGTOL), LSTOL(LLSTOL), ITMAX(IITMAX) {
		std::cout<<"FTOL: "<< ftol << std::endl;
		std::cout<<"GTOL: "<< GTOL << std::endl;
		std::cout<<"LSOTL: "<< LSTOL << std::endl;
	}
	
	Output_FrprmnV2 minimize(VecDoub_I &pp)
	{
		//const Int ITMAX=200;
		const Doub EPS=1.0e-18;
		//const Doub GTOL=1.0e-8;
		Doub gg,dgg,test;
		Int n=pp.size();
		p=pp;
		VecDoub g(n),h(n);
		xi.resize(n);
		Doub fp=func(p);
		func.df(p,xi);
		for (Int j=0;j<n;j++) {
			g[j] = -xi[j];
			xi[j]=h[j]=g[j];
		}
		for (Int its=0;its<ITMAX;its++) {
			iter=its;
			fret=linmin();
			if (2.0*abs(fret-fp) <= ftol*(abs(fret)+abs(fp)+EPS)) {
				std::cout << "Frprmn: converged iter="<< its 
					<<". ||psi'> - |psi>|^2 = "<< abs(fret-fp) << std::endl;
				return Output_FrprmnV2(its, fret, gg, p);
			}
			fp=fret;
			func.df(p,xi);
			test=0.0;
			Doub den=MAX(abs(fp),1.0);
			for (Int j=0;j<n;j++) {
				Doub temp=abs(xi[j])*MAX(abs(p[j]),1.0)/den;
				if (temp > test) test=temp;
			}
			if (test < GTOL) { 
				std::cout << "Frprmn: converged iter="<< its <<". g^2 = "<< test << std::endl;
				return Output_FrprmnV2(its, fret, gg, p);
			}
			dgg=gg=0.0;
			for (Int j=0;j<n;j++) {
				gg += g[j]*g[j];
			//	dgg += xi[j]*xi[j];
				dgg += (xi[j]+g[j])*xi[j];
			}
			if (gg == 0.0)
				return Output_FrprmnV2(its, fret, gg, p);
			Doub gam=dgg/gg;
			for (Int j=0;j<n;j++) {
				g[j] = -xi[j];
				xi[j]=h[j]=g[j]+gam*h[j];
			}
		}
		// throw("Too many iterations in frprmn");
		std::cout << "Frprmn: max iterations exceeded. g^2 = "<< test << std::endl;
		return Output_FrprmnV2(ITMAX, fret, gg, p);
	}
};

template <class T>
struct Frprmn : Linemethod<T> {
	Int iter;
	Doub fret;
	using Linemethod<T>::func;
	using Linemethod<T>::linmin;
	using Linemethod<T>::p;
	using Linemethod<T>::xi;
	const Doub ftol;
	const Int ITMAX;
	const Doub GTOL; 
	Frprmn(T &funcd, const Doub ftoll=3.0e-8, const Doub GGTOL=1.0e-8, const Int IITMAX=200)
	 : Linemethod<T>(funcd), ftol(ftoll), GTOL(GGTOL), ITMAX(IITMAX) {}
	VecDoub minimize(VecDoub_I &pp)
	{
		//const Int ITMAX=200;
		const Doub EPS=1.0e-18;
		//const Doub GTOL=1.0e-8;
		Doub gg,dgg,test;
		Int n=pp.size();
		p=pp;
		VecDoub g(n),h(n);
		xi.resize(n);
		Doub fp=func(p);
		func.df(p,xi);
		for (Int j=0;j<n;j++) {
			g[j] = -xi[j];
			xi[j]=h[j]=g[j];
		}
		for (Int its=0;its<ITMAX;its++) {
			iter=its;
			fret=linmin();
			if (2.0*abs(fret-fp) <= ftol*(abs(fret)+abs(fp)+EPS)) {
				std::cout << "Frprmn: converged iter="<< its 
					<<". ||psi'> - |psi>|^2 = "<< abs(fret-fp) << std::endl;
				return p;
			}
			fp=fret;
			func.df(p,xi);
			test=0.0;
			Doub den=MAX(abs(fp),1.0);
			for (Int j=0;j<n;j++) {
				Doub temp=abs(xi[j])*MAX(abs(p[j]),1.0)/den;
				if (temp > test) test=temp;
			}
			if (test < GTOL) { 
				std::cout << "Frprmn: converged iter="<< its <<". g^2 = "<< test << std::endl;
				return p;
			}
			dgg=gg=0.0;
			for (Int j=0;j<n;j++) {
				gg += g[j]*g[j];
			//	dgg += xi[j]*xi[j];
				dgg += (xi[j]+g[j])*xi[j];
			}
			if (gg == 0.0)
				return p;
			Doub gam=dgg/gg;
			for (Int j=0;j<n;j++) {
				g[j] = -xi[j];
				xi[j]=h[j]=g[j]+gam*h[j];
			}
		}
		// throw("Too many iterations in frprmn");
		std::cout << "Frprmn: max iterations exceeded. g^2 = "<< test << std::endl;
		return p;
	}
};